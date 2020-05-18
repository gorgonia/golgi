package golgi

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/qol"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

// WithClasses is a construction option that specifies how many classes are there in the embedding layer.
func WithClasses(classes int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *Embedding:
			l.classes = classes
			return layer, nil
		case Pass:
			return layer, nil
		default:
			return nil, errors.Errorf("WithClasses is a construction option that only supports Embedding. %T is not supported", layer)
		}
	}
}

// Embedding is a layer that represents an embedding layer.
//
// An embedding layer is essentially a matrix of the shape (classes, dimensions).
// While the Embedding layer can be done by means of using FC, this provides some ease of use.
//
// Specifically the Embedding layer supports forwarding of an input that is a slice of classes.
//
// Let us look at a word-based example, consider the vocab size to be the number of classes. The classical
// word embedding then is simply a (vocab, dims) matrix. Let's set the dims to be 50. For simplicity, let's set the vocab to 10.
// So that the embedding matrix W is a (10, 50) matrix.
//
//	W := ⸢w1_1  ...  w1_50⸣
//	             ⋮
//	     ⸤w10_1 ... w10_50⸥
//
// To select a word vector, we simply slice the matrix. For example, to get the vector of word ID 2, we slice W[2].
// This gives us a 50-dimension vector:
//
//	W[2] = [w2_1 ... w2_50]
//
// We can equally do this by multiplying the matrix with a one-hot vector. A vector O given as
// 	O := [0 0 1 0 0 0 0 0 0 0]
// when multiplied against W, will yield the same result as W[2].
//
// The usual way of selecting from a embedding matrix with a one-hot vector is quite cumbersome. This struct makes it easy.
//
// You can pass in a *tensor.Dense of qol.Class:
//
// 	wv := tensor.New(tensor.WithBacking([]qol.Class{4, 10, 0, 0, 0, 0, 0, 0,0,0}))
//	words := gorgonia.NewVector(g, gorgonia.WithShape(10), gorgonia.WithValue(wv))
// 	layer.Fwd(words)
//
// The Embedding layer's Fwd function will automatically transform a slice of classes into a one-hot matrix to be multiplied with.
type Embedding struct {
	w *G.Node

	// internal computation stuff

	// oh is a one hot vector/matrix used to "select" from w
	oh *G.Node

	// config

	// inputIsOneHot is used when the expected input is a onehot vector.
	inputIsOneHot bool

	// batch size
	bs int

	// size
	dims int

	// clases
	classes int

	// name
	name string

	// initialized
	initialized bool
}

func NewEmbedding(opts ...ConsOpt) *Embedding {
	retVal := new(Embedding)
	for _, opt := range opts {
		l, err := opt(retVal)
		if err != nil {
			panic(err)
		}
		retVal = l.(*Embedding)
	}
	if retVal.w != nil {
		retVal.initialized = true
	}
	return retVal
}

// Model returns the gorgonia.Nodes associated with the embedding layer.
func (l *Embedding) Model() G.Nodes { return G.Nodes{l.w} }

func (l *Embedding) Fwd(a G.Input) G.Result {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of Embedding %v", l.name))
	}
	shp := a.Node().Shape()
	if shp.Dims() > 2 {
		// error or reshape?
		// error for now:
		return G.Err(errors.Errorf("Cannot accept input of shape %v in Embedding", shp))
	}
	if shp[0] != l.bs {
		// error for now - in the next version of Gorgonia nodes are much lighter weight so we can fix this
		// TODO: FUTURE.
		return G.Err(errors.Errorf("Expected input to have batch size of %v. Got %v instead.", l.bs, shp[0]))
	}

	oh := a.Node()
	if !l.inputIsOneHot {
		oh = l.oh
		l.setOH(a.Node())
	}

	retVal, err := G.Mul(oh, l.w)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of Embedding %v - Mul error", l.name))
	}

	if !l.inputIsOneHot {
		switch shp.Dims() {
		case 2:
			// reshape result to (bs, dims)
			retVal, err = G.Reshape(retVal, tensor.Shape{l.bs, l.dims})
			if err != nil {
				return G.Err(errors.Wrapf(err, "Failed to reshape retVal  to (%v, %v)", l.bs, l.dims))
			}
		case 1:
			// NOOP
		case 0:
			// NOOP
		}
	}
	return retVal
}

func (l *Embedding) Name() string { return l.name }

func (l *Embedding) Shape() tensor.Shape { panic("NYi") }

func (l *Embedding) Type() hm.Type { panic("NYI") }

func (l *Embedding) Describe() {}

func (l *Embedding) Graph() *G.ExprGraph { return l.w.Graph() }

func (l *Embedding) IsInitialized() bool { return l.initialized }

// Init initializes the embedding layer.
func (l *Embedding) Init(xs ...*G.Node) (err error) {
	x := xs[0]
	g := x.Graph()
	of := x.Dtype()
	xshp := x.Shape()
	l.w = G.NewMatrix(g, of, G.WithShape(xshp[1], l.dims), G.WithInit(G.GlorotN(1)), G.WithName(l.name))

	if !l.inputIsOneHot {
		// we need to construct the one-hot matrix as well
		l.oh = G.NewMatrix(g, of, G.WithShape(xshp[0], l.classes), G.WithInit(G.Zeroes()), G.WithName(l.name+"dummy-1hot"))
	}
	return nil
}

// ConsEmbedding is a construction function to construct a *Embedding. This is typically used in a L() construction manner.
func ConsEmbedding(in G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	l := new(Embedding)
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*Embedding); !ok {
			return nil, errors.Errorf("Construction option for an embedding layer returned a non *Embedding. Got %T instead", o)
		}
	}

	if err := G.CheckOne(in); err != nil {
		return nil, errors.Wrapf(err, "Cons of an embedding layer %v", l.name)
	}
	x := in.Node()
	if err = l.Init(x); err != nil {
		return nil, err
	}
	return l, nil
}

// setOH is a function that sets the internal one hot vector/matrix
func (l *Embedding) setOH(a *G.Node) (err error) {
	T := a.Value().(*tensor.Dense)
	shp := T.Shape()
	T.Reshape(T.Shape().TotalSize())
	defer T.Reshape(shp...)

	var vec interface{}
	vec, err = native.Vector(T)
	if err != nil {
		return errors.Wrap(err, "Unable to set OneHot vector/matrix")
	}

	oh := l.oh.Value().(*tensor.Dense)

	// check shapes
	if oh.Shape().TotalSize() != T.Shape().TotalSize() {
		return errors.Errorf("Expected internal onehot vector/matrix to be %d. Got %d instead", T.Shape().TotalSize(), oh.Shape().TotalSize())
	}

	// reshape if necesary
	last := shp[len(shp)-1]
	newShape := tensor.Shape{oh.Shape().TotalSize() / last, last}
	oh.Reshape(newShape...)

	var classes []qol.Class
	switch v := vec.(type) {
	case []qol.Class:
		classes = v
	case []uint:
		classes := make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	case []int:
		classes := make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	case []float32:
		classes := make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	case []float64:
		classes := make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	}
	G.Let(l.oh, qol.UnsafeToOneHotMatrix(classes, uint(l.classes), oh))
	return nil
}
