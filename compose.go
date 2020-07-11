package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	_ Layer = (*Composition)(nil)
)

// Composition (∘) represents a composition of functions.
//
// The semantics of ∘(a, b)(x) is b(a(x)).
type Composition struct {
	a, b Term // can be thunk, Layer or *G.Node

	// store returns
	retVal   *G.Node
	retType  hm.Type
	retShape tensor.Shape
}

// Compose creates a composition of terms.
func Compose(a, b Term) (retVal *Composition) {
	if _, ok := a.(*G.Node); ok {
		a = nil
	}
	return &Composition{
		a: a,
		b: b,
	}
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
		l = Compose(l, next)
	}
	return l.(*Composition), nil
}

// Fwd runs the equation forwards
func (l *Composition) Fwd(a G.Input) (output G.Result) {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Forward of a Composition %v", l.Name()))
	}

	if l.retVal != nil {
		return l.retVal
	}
	input := a.Node()

	// apply a to input
	x, err := Apply(l.a, input)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Composition %v (a)", l.Name()))
	}
	if t, ok := x.(tag); ok {
		l.a = t.a.(Layer)
		x = t.b
	}

	// apply b to the result
	y, err := Apply(l.b, x)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Composition %v (b)", l.Name()))
	}
	switch yt := y.(type) {
	case tag:
		l.b = yt.a.(Layer)
		retVal, ok := yt.b.(*G.Node)
		if !ok {
			return G.Err(errors.Errorf("Error while forwarding Composition where layer is returned. Expected the result of a application to be a *Node. Got %v of %T instead", yt.b, yt.b))
		}
		return retVal
	case *G.Node:
		return yt
	default:
		return G.Err(errors.Errorf("Error while forwarding Composition. Expected the result of a application to be a *Node. Got %v of %T instead", yt, yt))
	}

}

// Model will return the gorgonia.Nodes associated with this composition
func (l *Composition) Model() (retVal G.Nodes) {
	if a, ok := l.a.(Layer); ok {
		return append(a.Model(), l.b.(Layer).Model()...)
	}
	return l.b.(Layer).Model()
}

// Name will return the name of the composition
func (l *Composition) Name() string { return fmt.Sprintf("%v ∘ %v", l.b, l.a) }

// Describe will describe a composition
func (l *Composition) Describe() { panic("STUB") }

// ByName returns a Term by name
func (l *Composition) ByName(name string) Term {
	if l.a == nil {
		goto next
	}
	if l.a.Name() == name {
		return l.a
	}
next:
	if l.b == nil {
		return nil
	}
	if l.b.Name() == name {
		return l.b
	}
	if bn, ok := l.a.(ByNamer); ok {
		if t := bn.ByName(name); t != nil {
			return t
		}
	}
	if bn, ok := l.b.(ByNamer); ok {
		if t := bn.ByName(name); t != nil {
			return t
		}
	}
	return nil
}

func (l *Composition) Graph() *G.ExprGraph {
	if gp, ok := l.a.(Grapher); ok {
		return gp.Graph()
	}
	if gp, ok := l.b.(Grapher); ok {
		return gp.Graph()
	}
	return nil
}

func (l *Composition) Runners() []Runner {
	var retVal []Runner
	if f, ok := l.a.(Runnerser); ok {
		retVal = append(retVal, f.Runners()...)
	}
	if f, ok := l.b.(Runnerser); ok {
		retVal = append(retVal, f.Runners()...)
	}
	return retVal
}
