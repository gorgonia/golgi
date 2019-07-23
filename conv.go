// +build !cuda

package golgi

import (
	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Conv struct{}

func (l *Conv) Model() G.Nodes {
	panic("not implemented")
}

func (l *Conv) Fwd(x Input) Result {
	panic("not implemented")
}

func (l *Conv) Type() hm.Type {
	panic("not implemented")
}

func (l *Conv) Shape() tensor.Shape {
	panic("not implemented")
}

func (l *Conv) Name() string {
	panic("not implemented")
}

func (l *Conv) Describe() {
	panic("not implemented")
}
