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

	// run A
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
		return G.Err(errors.Errorf("Fwd of Composition not handled for a of %T", l.a))
	}
next:

	if err != nil {
		return G.Err(errors.Wrapf(err, "Happened while doing `a` of Composition %v", l))
	}

	// run b
	var y G.Input
	switch bt := l.b.(type) {
	case *G.Node:
		y = bt
	case consThunk:
		if layer, err = bt.LayerCons(input, bt.Opts...); err != nil {
			return G.Err(errors.Wrapf(err, "Happned while calling the thunk of `b` of Join %v", l))
		}
		l.b = layer
		y = layer.Fwd(input)
	case Layer:
		y = bt.Fwd(input)
	default:
		return G.Err(errors.Errorf("Fwd of Join not handled for `b` of %T", l.b))
	}

	// check the results of a and b

	if err = G.CheckOne(x); err != nil {
		return G.Err(errors.Wrapf(err, "`a` of Join %v returned an error", l))
	}

	if err = G.CheckOne(y); err != nil {
		return G.Err(errors.Wrapf(err, "`b` of Join %v returned an error", l))
	}

	// perform the op

	xn := x.Node()
	yn := y.Node()

	switch l.op {
	case addOp:
		return G.LiftResult(G.Add(xn, yn))
	case elMulOp:
		return G.LiftResult(G.HadamardProd(xn, yn))
	}
	panic("Unreachable")
}
