package golgi

import (
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestCompose(t *testing.T) {
	softmax := func(a *gorgonia.Node) (*gorgonia.Node, error) { return gorgonia.SoftMax(a) }
	n := 100
	of := tensor.Float64
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, of, 4, gorgonia.WithName("X"), gorgonia.WithShape(n, 1, 28, 28), gorgonia.WithInit(gorgonia.GlorotU(1)))
	y := gorgonia.NewMatrix(g, of, gorgonia.WithName("Y"), gorgonia.WithShape(n, 10), gorgonia.WithInit(gorgonia.GlorotU(1)))
	nn, err := ComposeSeq(
		x,
		L(ConsReshape, ToShape(n, 784)),
		L(ConsFC, WithSize(50), WithName("l0"), AsBatched(true), WithActivation(gorgonia.Tanh), WithBias(true)),
		L(ConsDropout, WithProbability(0.5)),
		L(ConsFC, WithSize(150), WithName("l1"), AsBatched(true), WithActivation(gorgonia.Rectify)), // by default WithBias is true
		L(ConsLayerNorm, WithSize(20), WithName("Norm"), WithEps(0.001)),
		L(ConsFC, WithSize(10), WithName("l2"), AsBatched(true), WithActivation(softmax), WithBias(false)),
	)
	if err != nil {
		panic(err)
	}
	out := nn.Fwd(x)

	t.Logf("%v", nn.Name())
	t.Logf("%v", out)
	t.Logf("%v", y)

}
