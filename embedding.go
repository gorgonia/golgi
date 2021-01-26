package golgi

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/qol"
	"gorgonia.org/tensor"
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

// WithOneHotInput is a construction option for a *Embedding that specifiess the behaviour to accept one-hot-vector/matrix as input.
func WithOneHotInput() ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *Embedding:
			l.inputIsOneHot = true
			return layer, nil
		case Pass:
			return layer, nil
		default:
			return nil, errors.Errorf("WithOneHotInput is a construction option that is only supported by Embedding. %T is not supported", layer)
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

	// of
	of tensor.Dtype
}

// NewEmbedding creates a new embedding layer.
func NewEmbedding(opts ...ConsOpt) *Embedding {
	retVal := &Embedding{
		of: tensor.Float64, // default
		bs: 1,
	}

	for _, opt := range opts {
		l, err := opt(retVal)
		if err != nil {
			panic(err)
		}
		retVal, _ = l.(*Embedding)
	}
	if retVal.w != nil && (retVal.inputIsOneHot && retVal.oh != nil) || retVal.inputIsOneHot {
		retVal.initialized = true
	}
	if retVal.bs < 1 {
		retVal.bs = 1
	}
	return retVal
}

// Model returns the gorgonia.Nodes associated with the embedding layer.
func (l *Embedding) Model() G.Nodes { return G.Nodes{l.w} }

func (l *Embedding) Fwd(a G.Input) G.Result {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of Embedding %v", l.name))
	}
	if !l.initialized {
		if err := l.Init(a.Node()); err != nil {
			return G.Err(errors.Wrapf(err, "lazy initialization of %v failed", l.name))
		}
		l.initialized = true
	}

	shp := a.Node().Shape()
	if shp.Dims() > 2 {
		// error or reshape?
		// error for now:
		return G.Err(errors.Errorf("Cannot accept input of shape %v in Embedding", shp))
	}

	oh := a.Node()

	if !l.inputIsOneHot {
		oh = l.oh
		err := l.Run(a.Node())
		if err != nil {
			return G.Err(err)
		}
	}

	retVal, err := G.Mul(oh, l.w)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of Embedding %v - Mul error", l.name))
	}

	if !l.inputIsOneHot {
		switch shp.Dims() {
		case 2:
			// reshape result to (bs, dims)
			retVal, err = G.Reshape(retVal, tensor.Shape{shp[0], shp[1], l.dims})
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

func (l *Embedding) Describe() {}

func (l *Embedding) IsInitialized() bool { return l.initialized }

// Init initializes the embedding layer.
func (l *Embedding) Init(xs ...*G.Node) (err error) {
	x := xs[0]
	g := x.Graph()
	of := l.of

	if l.w == nil {
		l.w = G.NewMatrix(g, of, G.WithShape(l.classes, l.dims), G.WithInit(G.GlorotN(1)), G.WithName(l.name))
	}

	if !l.inputIsOneHot {
		// we need to construct the one-hot matrix as well
		l.oh = G.NewMatrix(g, of, G.WithShape(l.bs, l.classes), G.WithInit(G.Zeroes()), G.WithName(l.name+"dummy-1hot"))
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

// Graph returns the underlying computation graph. Embedding implements Grapher.
func (l *Embedding) Graph() *G.ExprGraph { return l.w.Graph() }

// Run is a function that sets the internal one hot vector/matrix
func (l *Embedding) Run(input G.Input) (err error) {
	if err := G.CheckOne(input); err != nil {
		return G.Err(errors.Wrapf(err, "Failed to run Embedding %v", l.name))
	}
	a := input.Node()
	T, _ := a.Value().(*tensor.Dense)

	vec := T.Data()

	oh, _ := l.oh.Value().(*tensor.Dense)

	var classes []qol.Class
	switch v := vec.(type) {
	case []qol.Class:
		classes = v
	case []uint:
		classes = make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	case []int:
		classes = make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	case []float32:
		classes = make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	case []float64:
		classes = make([]qol.Class, len(v))
		for i := range classes {
			classes[i] = qol.Class(v[i])
		}
	}

	return G.Let(l.oh, qol.UnsafeToOneHotMatrix(classes, uint(l.classes), oh))
}

// Runners returns the embedding itself
func (l *Embedding) Runners() []Runner { return []Runner{l} }
