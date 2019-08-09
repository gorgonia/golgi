package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Term interface {
	Type() hm.Type
}

// Name is a variable by name
type Name string

// Env is a linked list representing an environment.
// Within the documentation, an environment is written as such:
// 	e := (x ↦ X)
// `x` is the name while `X` is the *gorgonia.Node
//
// A longer environment may look like this:
//	e2 := (x ↦ X :: y ↦ Y)
//	                ^
// Here, e2 is pointing to the *Env that contains (y ↦ Y).
//
// When talking about Envs in general, it will often be written as a meta variable σ.
type Env struct {
	name string
	node *G.Node
	prev *Env
}

// NewEnv creates a new Env.
func NewEnv(name string, node *G.Node) *Env {
	return &Env{name: name, node: node}
}

// Extend allows users to extend the environment.
//
// Given an environment as follows:
// 	e := (x ↦ X)
// if `e.Extend(y, Y)` is called, the following is returned
//	e2 := (x ↦ X :: y ↦ Y)
//	                ^
// The pointer will be pointing to the *Env starting at y
func (e *Env) Extend(name string, node *G.Node) *Env {
	return &Env{name: name, node: node, prev: e}
}

// ByName returns the first node that matches the given name. It also returns the parent
//
// For example, if we have an Env as follows:
// 	e := (x ↦ X1 :: w ↦ W :: x ↦ X2)
// 	                         ^
//
// The caret indicates the pointer of *Env. Now, if e.ByName("x") is called,
// then the result returned will be X2 and (x ↦ X1 :: w ↦ W)
func (e *Env) ByName(name string) (*G.Node, *Env) {
	if e.name == name {
		return e.node, e.prev
	}
	if e.prev != nil {
		return e.prev.ByName(name)
	}
	return nil, nil
}

func (e *Env) Model() G.Nodes {
	retVal := G.Nodes{e.node}
	if e.prev != nil {
		retVal = append(retVal, e.prev.Model())
	}
	return retVal
}

func (e *Env) HintedModel(hint int) G.Nodes {
	prealloc := make(G.Nodes, 0, hint)
	e.hinted(prealloc)
	return prealloc
}

func (e *Env) hinted(prealloc G.Nodes) {
	prealloc = append(prealloc, e.node)
	if e.prev != nil {
		e.next.hinted(prealloc)
	}
}

type tag struct{ a, b Term }

// Layer represents a neural network layer.
// λ
type Layer interface {
	// σ - The weights are the "free variables" of a function
	Model() G.Nodes

	// Fwd represents the forward application of inputs
	// x.t
	Fwd(x G.Input) G.Result

	// meta stuff. This stuff is just placholder for more advanced things coming

	Term

	Shape() tensor.Shape

	// Name gives the layer's name
	Name() string

	// Serialization stuff

	// Describe returns the protobuf definition of a Layer that conforms to the ONNX standard
	Describe() // some protobuf things TODO
}

// Redefine redefines a layer with the given construction options. This is useful for re-initializing layers
func Redefine(l Layer, opts ...ConsOpt) (retVal Layer, err error) {
	for _, opt := range opts {
		if l, err := opt(l); err != nil {
			return l, err
		}
	}
	return l, nil
}

func Apply(a, b Term) (Term, error) {
	panic("STUBBED")
}

var (
	_ Layer = (*Composition)(nil)
)

// Composition represents a composition of functions
type Composition struct {
	a, b Term // can be thunk, Layer or *G.Node

	// store returns
	retVal   *G.Node
	retType  hm.Type
	retShape tensor.Shape
}

// Compose creates a composition of terms.
func Compose(a, b Term) (retVal *Composition, err error) {
	return &Composition{
		a: a,
		b: b,
	}, nil
}

// ComposeSeq creates a composition with the inputs written in left to right order
//
//
// The equivalent in F# is |>. The equivalent in Haskell is (flip (.))
func ComposeSeq(layers ...Term) (retVal *Composition, err error) {
	inputs := len(layers)
	switch inputs {
	case 0:
		return nil, errors.Errorf("Expected more than 1 input")
	case 1:
		// ?????
		return nil, errors.Errorf("Expected more than 1 input")
	}
	l := layers[0]
	for _, next := range layers[1:] {
		if l, err = Compose(l, next); err != nil {
			return nil, err
		}
	}
	return l.(*Composition), nil
}

func (l *Composition) Fwd(a G.Input) (output G.Result) {
	if err := G.CheckOne(a); err != nil {
		return G.Err{errors.Wrapf(err, "Forward of a Composition %v", l.Name())}
	}

	if l.retVal != nil {
		return l.retVal
	}
	input := a.Node()
	var x G.Input
	var layer Layer
	var err error
	switch at := l.a.(type) {
	case *G.Node:
		x = at
	case consThunk:
		if layer, err = at.LayerCons(input, at.Opts...); err != nil {
			goto next
		}
		l.a = layer
		x = layer.Fwd(input)
	case Layer:
		x = at.Fwd(input)
	default:
		return G.Err{errors.Errorf("Fwd of Composition not handled for a of %T", l.a)}
	}
next:
	if err != nil {
		return G.Err{errors.Wrapf(err, "Happened while doing a of Composition %v", l)}
	}

	switch bt := l.b.(type) {
	case *G.Node:
		return G.Err{errors.New("Cannot Fwd when b is a *Node")}
	case consThunk:
		if layer, err = bt.LayerCons(x.Node(), bt.Opts...); err != nil {
			return G.Err{errors.Wrapf(err, "Happened while doing b of Composition %v", l)}
		}
		l.b = layer
		output = layer.Fwd(x)
	case Layer:
		output = bt.Fwd(x)
	default:
		return G.Err{errors.Errorf("Fwd of Composition not handled for b of %T", l.b)}
	}
	return
}

func (l *Composition) Model() (retVal G.Nodes) {
	if a, ok := l.a.(Layer); ok {
		return append(a.Model(), l.b.(Layer).Model()...)
	}
	return l.b.(Layer).Model()
}

func (l *Composition) Type() hm.Type { return l.retType }

func (l *Composition) Shape() tensor.Shape { return l.retShape }

func (l *Composition) Name() string { return fmt.Sprintf("%v ∘ %v", l.b, l.a) }

func (l *Composition) Describe() { panic("STUB") }
