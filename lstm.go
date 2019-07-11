package golgi

import (
	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type LSTM struct{}

func (l *LSTM) Model() G.Nodes {
	panic("not implemented")
}

func (l *LSTM) Fwd(x *G.Node) (*G.Node, error) {
	panic("not implemented")
}

func (l *LSTM) Type() hm.Type {
	panic("not implemented")
}

func (l *LSTM) Shape() tensor.Shape {
	panic("not implemented")
}

func (l *LSTM) Name() string {
	panic("not implemented")
}

func (l *LSTM) Describe() {
	panic("not implemented")
}
