package golgi

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// WithWB is a FC specific construction option used to initialize a FC.
func WithWB(w, b *G.Node) ConsOpt {
	return func(layer Layer) (Layer, error) {
		fc, ok := layer.(*FC)
		if !ok {
			return layer, errors.Errorf("Expected a *FC. Got %v of %T instead", layer, layer)
		}
		fc.w = w
		fc.b = b
		fc.initialized = true
		return layer, nil
	}
}

// FC represents a fully connected layer
//
// If batched is set to true, then the first dimension is assumed to be the batch dimension
type FC struct {
	w, b *G.Node
	act  func(*G.Node) (*G.Node, error)

	name string

	// config
	size        int
	batched     bool
	nobias      bool
	initialized bool
}

// MakeFC creates a FC with the given parameters
func MakeFC(w, b *G.Node, act func(*G.Node) (*G.Node, error), name string, batched bool) FC {
	return FC{
		w:           w,
		b:           b,
		act:         act,
		name:        name,
		batched:     batched,
		initialized: true,
	}
}

// NewFC is the usual way to create a FC
func NewFC(opts ...ConsOpt) *FC {
	retVal := new(FC)
	for _, opt := range opts {
		l, err := opt(retVal)
		if err != nil {
			panic(err)
		}
		retVal = l.(*FC)
	}
	retVal.initialized = true
	return retVal
}

func (l *FC) Model() G.Nodes {
	if l.nobias {
		return G.Nodes{l.w}
	}
	return G.Nodes{l.w, l.b}
}

func (l *FC) Fwd(x *G.Node) (*G.Node, error) {
	var xw, xwb *G.Node
	var err error
	if xw, err = G.Mul(x, l.w); err != nil {
		return nil, err
	}

	if l.b == nil {
		xwb = xw
		goto act
	}

	if l.batched && !(l.b.Shape().Eq(xw.Shape())) {
		if xwb, err = G.BroadcastAdd(xw, l.b, nil, []byte{0}); err != nil {
			return nil, err
		}
	} else {
		if xwb, err = G.Add(xw, l.b); err != nil {
			return nil, err
		}
	}
act:
	if l.act == nil {
		return xwb, nil
	}
	return l.act(xwb)
}

func (l *FC) Type() hm.Type       { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b')) }
func (l *FC) Shape() tensor.Shape { return l.b.Shape() }
func (l *FC) Name() string        { return l.name }
func (l *FC) Describe()           { panic("STUB") }

func ConsFC(x *G.Node, opts ...ConsOpt) (retVal Layer, err error) {
	inshape := x.Shape()
	if inshape.Dims() > 2 || inshape.Dims() == 0 {
		return nil, errors.Errorf("Expected shape is either a vector or a matrix")
	}

	// construct
	l := &FC{}
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*FC); !ok {
			return nil, errors.Errorf("Construction Option returned a non FC. Got %T instead", l)
		}
	}

	// prep
	g := x.Graph()
	of := x.Dtype()
	X := x
	if x.IsVec() {
		X, err = G.Reshape(x, tensor.Shape{1, x.Shape()[0]})
	}
	xshp := X.Shape()

	l.w = G.NewMatrix(g, of, G.WithShape(xshp[1], l.size), G.WithInit(G.GlorotU(1)), G.WithName(l.name+"_W"))
	switch {
	case l.batched && !l.nobias:
		l.b = G.NewMatrix(g, of, G.WithShape(1, l.size), G.WithInit(G.Zeroes()), G.WithName(l.name+"_B"))
	case !l.batched && !l.nobias:
		l.b = G.NewMatrix(g, of, G.WithShape(xshp[0], xshp[1]), G.WithInit(G.Zeroes()), G.WithName(l.name+"_B"))
	}

	return l, nil
}
