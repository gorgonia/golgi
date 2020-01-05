package golgi

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Metadata is not a real Layer. Its main aims is to extract metadata such as name or size from ConsOpts. This is useful in cases where the metadata needs to be composed as well.
// Note that the fields may end up being all empty.
type Metadata struct {
	name         string
	Size         int
	shape        tensor.Shape
	ActivationFn func(*G.Node) (*G.Node, error)

	//internal state
	upd uint // counts the number of times the data structure has been updated.
}

// Name returns the name. Conveniently, this makes *Metadata fulfil the Layer interface, so we may use it to extract the desired metadata.
// Unfortunately this also means that the name is not an exported field. A little inconsistency there.
func (m *Metadata) Name() string { return m.name }

func (m *Metadata) Shape() tensor.Shape { return m.shape }

func (m *Metadata) Describe()              {}
func (m *Metadata) Model() G.Nodes         { return nil }
func (m *Metadata) Fwd(x G.Input) G.Result { return G.Err(errors.New("Metadata is a dummy Layer")) }
func (m *Metadata) Type() hm.Type          { return nil }

// SetName allows for names to be set by a ConsOpt
func (m *Metadata) SetName(name string) error {
	if m.name != "" {
		return errors.Errorf("A name exists - %q ", m.name)
	}
	m.name = name
	m.upd++
	return nil
}

// SetSize allows for the metadata struct to be filled by a ConsOpt
func (m *Metadata) SetSize(size int) error {
	if m.Size != 0 {
		return errors.Errorf("A clashing size %d exists.", m.Size)
	}
	m.Size = size
	m.upd++
	return nil
}

// SetActivationFn allows the metadata to store activation function.
func (m *Metadata) SetActivationFn(act func(*G.Node) (*G.Node, error)) error {
	if m.ActivationFn != nil {
		return errors.New("A clashing activation function already exists")
	}
	m.ActivationFn = act
	m.upd++
	return nil
}

// ExtractMetadata extracts common metadata from a list of ConsOpts. It returns the metadata. Any unused ConsOpt is also returned.
// This allows users to selectively use the metadata and/or ConsOpt options
func ExtractMetadata(opts ...ConsOpt) (retVal Metadata, unused []ConsOpt, err error) {
	var l Layer = &retVal
	var m *Metadata = &retVal
	var ok bool
	upd := m.upd
	for _, opt := range opts {
		if l, err = opt(l); err != nil {
			return Metadata{}, unused, err
		}
		if m, ok = l.(*Metadata); !ok {
			return Metadata{}, unused, errors.Errorf("ConsOpt mutated metadata. Got %T instead", l)
		}
		if m.upd > upd {
			upd = m.upd
		} else {
			unused = append(unused, opt)
		}
	}
	return *m, unused, nil
}

// ConsOpt is a construction option for layers
type ConsOpt func(Layer) (Layer, error)

type namesetter interface {
	SetName(a string) error
}

type sizeSetter interface {
	SetSize(int) error
}

type actSetter interface {
	SetActivationFn(act func(*G.Node) (*G.Node, error)) error
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
		case *layerNorm:
			l.name = name
			return layer, nil
		case *LSTM:
		case *Conv:
		case unnameable:
			return layer, nil
		case namesetter:
			err := l.SetName(name)
			return layer, err
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
		case sizeSetter:
			l.SetSize(size[0])
			return layer, nil
		}
		return nil, errors.Errorf("WithSize Unhandled Layer type: %T", layer)
	}
}

// WithActivation sets the activation function of a layer
func WithActivation(act func(*G.Node) (*G.Node, error)) ConsOpt {
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
		}
		return nil, errors.Errorf("WithActivation Unhandled Layer type: %T", layer)
	}
}

// ToShape is a ConsOpt for Reshape only.
func ToShape(shp ...int) ConsOpt {
	return func(layer Layer) (Layer, error) {
		if _, ok := layer.(reshape); ok {
			return reshape(tensor.Shape(shp)), nil
		}
		return nil, errors.Errorf("ToShape Unhandled Layer type: %T", layer)
	}
}

// WithProbability is a ConsOpt for Dropout only.
func WithProbability(prob float64) ConsOpt {
	return func(layer Layer) (Layer, error) {
		if _, ok := layer.(dropout); ok {
			return dropout(prob), nil
		}
		return nil, errors.Errorf("WithProbability Unhandled Layer type: %T", layer)
	}
}

// WithEps is a ConsOpt for constructing Layer Norms only.
func WithEps(eps float64) ConsOpt {
	return func(layer Layer) (Layer, error) {
		if l, ok := layer.(*layerNorm); ok {
			l.eps = eps
			return l, nil
		}
		return nil, errors.Errorf("WithEps Unhandled Layer type: %T", layer)
	}
}
