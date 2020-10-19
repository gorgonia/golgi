package golgi

import (
	"math"

	"github.com/chewxy/math32"
	G "gorgonia.org/gorgonia"
)

// Activation represents the types of ActivationFunctions that Golgi understands.
//
// The Activation type is useful as it allows the models to be serialized (you cannot serialize ActivationFunction).
// Use ActivationMap() to get the relevant ActivationFunction.
type Activation int

const (
	Identity Activation = iota
	Sigmoid
	Tanh
	ReLU
	GeLU
	LeakyReLU
	ELU
	Cube
	SoftMax
)

var maxact = Cube

var internalmaps = map[Activation]ActivationFunction{
	Identity:  nil,
	Sigmoid:   G.Sigmoid,
	Tanh:      G.Tanh,
	ReLU:      G.Rectify,
	GeLU:      GeLUFn, // TODO
	LeakyReLU: nil,    // TODO
	ELU:       nil,    //TODO
	Cube:      G.Cube,
	SoftMax:   SoftMaxFn,
}

// ActivationMap is a map from Activation to ActivationFunction. The mapping function is finite. If an invalid Activation is passed in, nil will be returned.
func ActivationMap(a Activation) ActivationFunction { return internalmaps[a] }

var elmul = G.Lift2(G.HadamardProd)
var tanh = G.Lift1(G.Tanh)
var add = G.Lift2(G.Add)
var mul = G.Lift2(G.Mul)
var cube = G.Lift1(G.Cube)

// GeLUFn is an activation function. See https://arxiv.org/abs/1606.08415.
func GeLUFn(a *G.Node) (*G.Node, error) {
	var half, magic, sqrt2Overπ, one *G.Node
	t := a.Dtype()
	switch t {
	case G.Float64:
		half = G.NewConstant(0.5)
		magic = G.NewConstant(0.044715)
		one = G.NewConstant(1.0)
		sqrt2Overπ = G.NewConstant(math.Sqrt(2.0 / math.Pi))
	case G.Float32:
		half = G.NewConstant(float32(0.5))
		magic = G.NewConstant(float32(0.044715))
		one = G.NewConstant(float32(1.0))
		sqrt2Overπ = G.NewConstant(math32.Sqrt(float32(2.0) / math32.Pi))
	}
	// TODO:  propoer if res, err := ...
	retVal := elmul(
		elmul(half, a),
		add(
			one,
			tanh(mul(
				sqrt2Overπ,
				add(
					a,
					mul(magic, cube(a)),
				),
			)),
		),
	)
	return retVal.Node(), retVal.Err()
}

// SoftMaxFn implements softmax without axis
func SoftMaxFn(a *G.Node) (*G.Node, error) {
	return G.SoftMax(a)
}
