// +build !cuda

package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ConsMaxPool is a MaxPool construction function. It takes a gorgonia.Input that has a *gorgonia.Node.
func ConsMaxPool(in gorgonia.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()
	if x == nil {
		return nil, fmt.Errorf("ConsMaxPool expects a *Node. Got input %v of  %T instead", in, in)
	}

	inshape := x.Shape()
	if inshape.Dims() != 4 || inshape.Dims() == 0 {
		return nil, fmt.Errorf("Expected shape is a matrix")
	}

	l := &MaxPool{
		kernelShape: tensor.Shape{2, 2},
		pad:         []int{0, 0},
		stride:      []int{2, 2},
	}

	for _, opt := range opts {
		var (
			o  Layer
			ok bool
		)

		if o, err = opt(l); err != nil {
			return nil, err
		}

		if l, ok = o.(*MaxPool); !ok {
			return nil, fmt.Errorf("Construction Option returned a non MaxPool. Got %T instead", o)
		}
	}

	// prep
	if err = l.Init(x); err != nil {
		return nil, err
	}

	return l, nil
}

// Init will initialize the fully connected layer
func (l *MaxPool) Init(xs ...*gorgonia.Node) (err error) {
	l.initialized = true

	return nil
}

// MaxPool represents a MaxPoololution layer
type MaxPool struct {
	name string
	size int

	kernelShape tensor.Shape
	pad, stride []int

	// optional config
	dropout *float64 // nil when shouldn't be applied

	initialized bool
}

// SetSize sets the size of the layer
func (l *MaxPool) SetSize(s int) error {
	l.size = s
	return nil
}

// SetName sets the name of the layer
func (l *MaxPool) SetName(n string) error {
	l.name = n
	return nil
}

// SetDropout sets the dropout of the layer
func (l *MaxPool) SetDropout(d float64) error {
	l.dropout = &d
	return nil
}

// Model will return the gorgonia.Nodes associated with this MaxPoololution layer
func (l *MaxPool) Model() gorgonia.Nodes {
	return gorgonia.Nodes{}
}

// Fwd runs the equation forwards
func (l *MaxPool) Fwd(x gorgonia.Input) gorgonia.Result {
	if err := gorgonia.CheckOne(x); err != nil {
		return gorgonia.Err(fmt.Errorf("Fwd of MaxPool %v: %w", l.name, err))
	}

	result, err := gorgonia.MaxPool2D(x.Node(), l.kernelShape, l.pad, l.stride)
	if err != nil {
		return wrapErr(l, "applying max pool to %v: %w", x.Node().Shape(), err)
	}

	if l.dropout != nil {
		result, err = gorgonia.Dropout(result, *l.dropout)
		if err != nil {
			return gorgonia.Err(err)
		}
	}

	logf("%T shape %s: %v", l, l.name, result.Shape())

	return result
}

// Type will return the hm.Type of the MaxPoololution layer
func (l *MaxPool) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Name will return the name of the MaxPoololution layer
func (l *MaxPool) Name() string {
	return l.name
}

// Describe will describe a MaxPoololution layer
func (l *MaxPool) Describe() {
	panic("not implemented")
}

var (
	_ sizeSetter      = &MaxPool{}
	_ namesetter      = &MaxPool{}
	_ dropoutConfiger = &MaxPool{}
)
