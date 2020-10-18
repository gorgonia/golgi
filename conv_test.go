package golgi

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestConvNet(t *testing.T) {
	c := require.New(t)

	bs := 32
	convKS := tensor.Shape{3, 3}
	mpKS := tensor.Shape{2, 2}

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithName("x"), gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithInit(gorgonia.GlorotU(1)))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithName("y"), gorgonia.WithShape(bs, 10), gorgonia.WithInit(gorgonia.GlorotU(1)))

	nn, err := ComposeSeq(
		x,
		L(ConsConv, WithName("layer 0"), WithSize(bs, 1), WithKernelShape(convKS)),
		L(ConsMaxPool, WithName("layer 0"), WithKernelShape(mpKS)),
		L(ConsDropout, WithName("layer 0"), WithProbability(0.2)),
		L(ConsConv, WithName("layer 1"), WithSize(bs*2, bs), WithKernelShape(convKS)),
		L(ConsMaxPool, WithName("layer 1"), WithKernelShape(mpKS)),
		L(ConsDropout, WithName("layer 1"), WithProbability(0.2)),
		L(ConsConv, WithName("layer 2"), WithSize(bs*4, bs*2), WithKernelShape(convKS)),
		L(ConsMaxPool, WithName("layer 2"), WithKernelShape(mpKS)),
		L(ConsReshape, WithName("layer 2"), ToShape(bs, (bs*4)*3*3)),
		L(ConsDropout, WithName("layer 2"), WithProbability(0.2)),
		L(ConsFC, WithName("layer 3"), WithSize(625), WithActivation(gorgonia.Rectify)),
		L(ConsDropout, WithName("layer 3"), WithProbability(0.55)),
		L(ConsFC, WithName("output"), WithSize(10), WithActivation(SoftMaxFn)),
	)

	c.NoError(err)

	out := nn.Fwd(x)

	err = gorgonia.CheckOne(out)
	c.NoError(err)

	losses := gorgonia.Must(RMS(out, y))
	model := nn.Model()

	_, err = gorgonia.Grad(losses, model...)
	c.NoError(err)

	var costVal gorgonia.Value
	gorgonia.Read(losses, &costVal)

	m := gorgonia.NewTapeMachine(g)

	err = m.RunAll()
	c.NoError(err)

	cost, ok := costVal.Data().(float64)
	c.True(ok)
	c.NotZero(cost)
}
