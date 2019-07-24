package golgi

import (
	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type unnameable interface {
	unnamed()
}

type id struct{}

func (l id) Model() G.Nodes       { return nil }
func (l id) Fwd(x Input) G.Result { return x.Node() }
func (l id) Type() hm.Type        { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a')) }
func (l id) Shape() tensor.Shape  { panic("not implemented") }
func (l id) Name() string         { return "I" }
func (l id) Describe()            {}
func (l id) unnamed()             {}

type k struct{ *G.Node }

func (l k) Model() G.Nodes       { return nil }
func (l k) Fwd(x Input) G.Result { return l.Node }
func (l k) Type() hm.Type        { return hm.NewFnType(hm.TypeVariable('a'), l.Node.Type()) }
func (l k) Shape() tensor.Shape  { panic("not implemented") }
func (l k) Name() string         { return "K" }
func (l k) Describe()            {}
func (l k) unnamed()             {}

type reshape tensor.Shape

func ConsReshape(x *G.Node, opts ...ConsOpt) (l Layer, err error) {
	l = reshape(nil)
	for _, opt := range opts {
		if l, err = opt(l); err != nil {
			return nil, err
		}
	}
	return l, nil
}

func (l reshape) Model() G.Nodes { return nil }
func (l reshape) Fwd(x Input) G.Result {
	if err := G.CheckOne(x); err != nil {
		return G.Err{err}
	}
	return G.LiftResult(gorgonia.Reshape(x.Node(), tensor.Shape(l)))
}
func (l reshape) Type() hm.Type       { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a')) }
func (l reshape) Shape() tensor.Shape { return tensor.Shape(l) }
func (l reshape) Name() string        { return "" }
func (l reshape) Describe()           {}
func (l reshape) unnamed()            {}

type dropout float64

func ConsDropout(x *G.Node, opts ...ConsOpt) (l Layer, err error) {
	l = dropout(0)
	for _, opt := range opts {
		if l, err = opt(l); err != nil {
			return nil, err
		}
	}
	return l, nil
}

func (l dropout) Model() G.Nodes { return nil }
func (l dropout) Fwd(x Input) G.Result {
	if err := G.CheckOne(x); err != nil {
		return G.Err{err}
	}
	return G.LiftResult(gorgonia.Dropout(x.Node(), float64(l)))
}
func (l dropout) Type() hm.Type       { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a')) }
func (l dropout) Shape() tensor.Shape { panic("not implemented") }
func (l dropout) Name() string        { return "" }
func (l dropout) Describe()           {}
func (l dropout) unnamed()            {}
