package golgi

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	_ Layer = (*layerNorm)(nil)
)

// layerNorm performs layer normalization as per https://arxiv.org/abs/1607.06450
type layerNorm struct {
	FC
	epsNode *G.Node
	eps     float64

	flops        int
	computeFLOPs bool
}

// There is no Model() method. When Model() is called, it simply calls the FC's Model()
// There is no Type() method
// There is no Shape() method

// NewLayerNorm creates a layer-normalization layer. It does not initialize the layer.
func NewLayerNorm(opts ...ConsOpt) Layer {
	l := &layerNorm{
		eps: 1e-5,
	}
	for _, opt := range opts {
		var o Layer
		var err error
		if o, err = opt(l); err != nil {
			panic(err)
		}
		l = o.(*layerNorm) // panics if not layernorm
	}
	// misc settings that has to be reset in case anything else gets set
	l.batched = true
	l.act = nil
	l.nobias = false
	return l
}

// ConsLayerNorm is a construction function for a layer normalization layer. `in` has to be at least a *gorgonia.Node
func ConsLayerNorm(in G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()

	inshape := x.Shape()
	if inshape.Dims() > 2 || inshape.Dims() == 0 {
		return nil, errors.Errorf("Expected shape is either a vector or a matrix")
	}

	// construct
	l := &layerNorm{
		eps: 1e-5,
	}
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*layerNorm); !ok {
			return nil, errors.Errorf("Construction Option returned a non layerNorm. Got %T instead", l)
		}
	}
	// misc settings that has to be reset in case anything else gets set
	l.batched = true
	l.act = nil
	l.nobias = false

	if err = l.Init(x); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *layerNorm) Fwd(a G.Input) G.Result {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrap(err, "Fwd of layer norm failed."))
	}

	x := a.Node()
	xshp := x.Shape()
	last := xshp.Dims() - 1

	// lazy initialization
	if !l.IsInitialized() {
		if err := l.Init(x); err != nil {
			return G.Err(errors.Wrapf(err, "Lazy initialization of *layerNorm %v", l.name))
		}
	}

	var err error
	var μ, xmμ, σ2, sd, newX *G.Node
	if μ, err = G.KeepDims(x, false, func(x *G.Node) (*G.Node, error) { return G.Mean(x, last) }); err != nil {
		return G.Err(errors.Wrapf(err, "Unable to find mean of %dth dimension of %v", last, x))
	}
	// xmu: x-μ
	if xmμ, err = G.BroadcastSub(x, μ, nil, []byte{byte(last)}); err != nil {
		return G.Err(errors.Wrapf(err, "Unable to perform (x-μ). Shapes - x: %v,  μ: %v. Broadcast on right axis: %v", x.Shape(), μ.Shape(), last))
	}

	// σ2: ((x-μ)^2)/N
	if σ2, err = G.Square(xmμ); err != nil {
		return G.Err(errors.Wrap(err, "Unable to perform (x-μ)^2"))
	}
	if σ2, err = G.KeepDims(σ2, false, func(x *G.Node) (*G.Node, error) { return G.Mean(x, last) }); err != nil {
		return G.Err(errors.Wrap(err, "Unable to calculate Mean Squared Variance"))
	}

	// purturb the variance before sqrting it
	if sd, err = G.Add(σ2, l.epsNode); err != nil {
		return G.Err(errors.Wrap(err, "Unable to purturb the variance"))
	}
	if sd, err = G.Sqrt(sd); err != nil {
		return G.Err(errors.Wrap(err, "Unable to sqrt the variance"))
	}

	// now we have a new x
	if newX, err = G.BroadcastHadamardDiv(xmμ, sd, nil, []byte{byte(last)}); err != nil {
		return G.Err(errors.Wrapf(err, "Unable to do (x-μ)/σ. Shapes - xmμ: %v, sd: %v. Broadcast on right axis: %v", xmμ.Shape(), sd.Shape(), last))
	}

	// the rest is straightforwards FC
	return l.FC.Fwd(newX)
}

func MakeLayerNorm(opts ...ConsOpt) Layer {
	l := &layerNorm{
		eps: 1e-5,
	}
	for _, opt := range opts {
		var o Layer
		var err error
		if o, err = opt(l); err != nil {
			panic(err)
		}
		l = o.(*layerNorm) // panics if not layernorm
	}
	// misc settings that has to be reset in case anything else gets set
	l.batched = true
	l.act = nil
	l.nobias = false
	if l.FC.w != nil || l.FC.b != nil {
		l.FC.initialized = true
	}
	return l
}

func (l *layerNorm) Init(xs ...*G.Node) (err error) {
	x := xs[0]
	// prep
	g := x.Graph()
	of := x.Dtype()
	X := x
	if x.IsVec() {
		X, err = G.Reshape(x, tensor.Shape{1, x.Shape()[0]})
		if err != nil {
			return errors.Wrapf(err, "While initializing layerNorm")
		}
	}
	xshp := X.Shape()

	switch of {
	case tensor.Float32:
		l.epsNode = G.NewConstant(float32(l.eps))
	case tensor.Float64:
		l.epsNode = G.NewConstant(l.eps)
	default:
		return errors.New("Layer Norm only supports Float32 or Float64")
	}
	l.w = G.NewMatrix(g, of, G.WithShape(xshp[1], l.size), G.WithInit(G.Ones()), G.WithName(l.name+"_W"))
	l.b = G.NewMatrix(g, of, G.WithShape(1, l.size), G.WithInit(G.Zeroes()), G.WithName(l.name+"_B"))
	l.initialized = true
	if l.computeFLOPs {
		l.flops = l.doComputeFLOPs(X.Shape())
	}

	return nil
}

func (l *layerNorm) SetComputeFLOPs(toCompute bool) error {
	l.computeFLOPs = toCompute
	l.FC.computeFLOPs = toCompute
	return nil
}

func (l *layerNorm) doComputeFLOPs(input tensor.Shape) int {
	mean := input.TotalSize()            // x-μ
	meanSq := mean * 2                   // (x-μ)^2
	variance := meanSq + mean            // (x-μ)^2 / N
	variancePerturbed := variance + mean // perturbation
	sqrt := variancePerturbed + mean     // sqrt
	div := sqrt + mean
	fc := l.FC.doComputeFLOPs(input)
	return div + fc
}
