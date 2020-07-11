package golgi

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
)

var (
	_ Layer = (*Join)(nil)
)

type joinOp int

const (
	composeOp joinOp = iota
	addOp
	elMulOp
)

// Joins are generalized compositions.
type Join struct {
	Composition
	op joinOp
}

// Add adds the results of two layers/terms.
func Add(a, b Term) *Join {
	return &Join{
		Composition: Composition{
			a: a,
			b: b,
		},
		op: addOp,
	}
}

// HadamardProd performs a elementwise multiplicatoin on the results of two layers/terms.
func HadamardProd(a, b Term) *Join {
	return &Join{
		Composition: Composition{
			a: a,
			b: b,
		},
		op: elMulOp,
	}
}

// Fwd runs the equation forwards.
func (l *Join) Fwd(a G.Input) (output G.Result) {
	if l.op == composeOp {
		return l.Composition.Fwd(a)
	}

	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Forward of a Join %v", l.Name()))
	}

	if l.retVal != nil {
		return l.retVal
	}
	input := a.Node()

	x, err := Apply(l.a, input)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Join %v - Applying %v to %v failed", l.Name(), l.a, input.Name()))
	}
	xn, ok := x.(*G.Node)
	if !ok {
		return G.Err(errors.Errorf("Expected the result of applying %v to %v to return a *Node. Got %v of %T instead", l.a, input.Name(), x, x))
	}

	y, err := Apply(l.b, input)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Join %v - Applying %v to %v failed", l.Name(), l.b, input.Name()))
	}
	yn, ok := y.(*G.Node)
	if !ok {
		return G.Err(errors.Errorf("Expected the result of applying %v to %v to return a *Node. Got %v of %T instead", l.a, input.Name(), y, y))
	}

	// perform the op

	switch l.op {
	case addOp:
		return G.LiftResult(G.Add(xn, yn))
	case elMulOp:
		return G.LiftResult(G.HadamardProd(xn, yn))
	}
	panic("Unreachable")
}
