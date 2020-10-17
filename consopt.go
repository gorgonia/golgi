package golgi

import (
	"fmt"

	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ConsOpt is a construction option for layers
type ConsOpt func(Layer) (Layer, error)

// ReshapeFn defines a function to reshape a tensor
type ReshapeFn func(s tensor.Shape) tensor.Shape

type namesetter interface {
	SetName(a string) error
}

type sizeSetter interface {
	SetSize(int) error
}

type actSetter interface {
	SetActivationFn(act ActivationFunction) error
}

type dropoutConfiger interface {
	SetDropout(prob float64) error
}

// WithName creates a layer that is named.
//
// If the layer is unnameable (i.e. trivial layers), then there is no effect.
func WithName(name string) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.name = name
			return layer, nil
		case *Embedding:
			l.name = name
			return layer, nil
		case *layerNorm:
			l.name = name
			return layer, nil
		case *LSTM:
		case unnameable:
			return layer, nil
		case namesetter:
			err := l.SetName(name)
			return layer, err
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithName Unhandled Layer type: %T", layer)
	}
}

// AsBatched defines a layer that performs batched operation or not.
// By default most layers in this package are batched.
func AsBatched(batched bool) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.batched = batched
			return layer, nil
		case *LSTM:
		case *Conv:
		case reshape:
			return layer, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("AsBatched Unhandled Layer type: %T", layer)
	}
}

// WithBias defines a layer with or without a bias
func WithBias(withbias bool) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.nobias = !withbias
			return layer, nil
		}
		return layer, nil
	}
}

// WithSize automatically creates a layer of the given sizes.
func WithSize(size ...int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		// NO RESHAPE ALLOWED
		switch l := layer.(type) {
		case *FC:
			l.size = size[0]
			return l, nil
		case *Embedding:
			l.dims = size[0]
			return l, nil
		case sizeSetter:
			l.SetSize(size[0])
			return layer, nil
		case Pass:
			return layer, nil
		case *Conv:
			l.SetSize(size...)
			return layer, nil
		case *LSTM:
			l.size = size[0]
			return l, nil
		}

		return nil, errors.Errorf("WithSize Unhandled Layer type: %T", layer)
	}
}

// WithBatchSize creates a layer with a given batch size.
func WithBatchSize(bs int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *Embedding:
			l.bs = bs
			return l, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithBatchSize does not handle Layer type %T", layer)
	}
}

// WithActivation sets the activation function of a layer
func WithActivation(act ActivationFunction) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.act = act
			return layer, nil
		case reshape:
			return layer, nil
		case actSetter:
			l.SetActivationFn(act)
			return layer, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithActivation Unhandled Layer type: %T", layer)
	}
}

func Of(dt tensor.Dtype) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *Embedding:
			l.of = dt
			return layer, nil
		case Pass:
			return layer, nil
		default:

			return nil, errors.Errorf("Of does not yet support Layer type %T", layer)
		}
	}
}

// ToShape is a ConsOpt for Reshape only.
func ToShape(shp ...int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch layer.(type) {
		case reshape:
			return reshape(tensor.Shape(shp)), nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("ToShape Unhandled Layer type: %T", layer)
	}
}

// WithProbability is a ConsOpt for Dropout only.
func WithProbability(prob float64) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case dropout:
			return dropout(prob), nil
		case dropoutConfiger:
			err := l.SetDropout(prob)
			if err != nil {
				return nil, err
			}
			return layer, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithProbability Unhandled Layer type: %T", layer)
	}
}

// WithEps is a ConsOpt for constructing Layer Norms only.
func WithEps(eps float64) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *layerNorm:
			l.eps = eps
			return l, nil
		case Pass:
			return layer, nil
		}
		return nil, errors.Errorf("WithEps Unhandled Layer type: %T", layer)
	}
}

// WithConst is a construction option for the skip Layer
func WithConst(c *G.Node) ConsOpt {
	return func(l Layer) (Layer, error) {
		if a, ok := l.(*skip); ok {
			a.b = c
			return l, nil
		}
		return nil, errors.Errorf("WithConst expects a *skip. Got %T instead", l)
	}
}

// WithWeights constructs a layer with the given weights.
func WithWeights(w *G.Node) ConsOpt {
	return func(layer Layer) (Layer, error) {
		switch l := layer.(type) {
		case *FC:
			l.w = w
			l.initialized = true
		case *Embedding:
			l.w = w
			// l.initialized = true
			// this cannot be true unless l.oh has been set.
		default:
			return nil, errors.Errorf("WithWeights does not handle layer of type %T", layer)
		}
		return layer, nil
	}
}

// WithKernelShape sets the kernel shape of the layer
func WithKernelShape(s tensor.Shape) ConsOpt {
	return func(l Layer) (Layer, error) {
		switch c := l.(type) {
		case *Conv:
			c.kernelShape = s

			return c, nil
		case *MaxPool:
			c.kernelShape = s

			return c, nil
		}

		return nil, fmt.Errorf("Setting kernel shape is not supported by this layer: %T", l)
	}
}

// WithPad sets the pad of the layer
func WithPad(p []int) ConsOpt {
	return func(l Layer) (Layer, error) {
		switch c := l.(type) {
		case *Conv:
			c.pad = p

			return c, nil
		}

		return nil, fmt.Errorf("Setting pad is not supported by this layer")
	}
}

// WithStride sets the stride of the layer
func WithStride(s []int) ConsOpt {
	return func(l Layer) (Layer, error) {
		switch c := l.(type) {
		case *Conv:
			c.stride = s

			return c, nil
		}

		return nil, fmt.Errorf("Setting stride is not supported by this layer")
	}
}

// WithDilation sets the dilation of the layer
func WithDilation(s []int) ConsOpt {
	return func(l Layer) (Layer, error) {
		switch c := l.(type) {
		case *Conv:
			c.dilation = s

			return c, nil
		}

		return nil, fmt.Errorf("Setting dilation is not supported by this layer")
	}
}
