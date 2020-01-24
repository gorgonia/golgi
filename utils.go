package golgi

import "gorgonia.org/gorgonia"

// ActivationFunction represents an activation function
// Note: This may become an interface once we've worked through all the linter errors
type ActivationFunction func(*gorgonia.Node) (*gorgonia.Node, error)

// BroadcastAdd will perform a broadcast addtion
func BroadcastAdd(a, b *gorgonia.Node, left, right []byte) (*gorgonia.Node, error) {
	if a.Shape().Eq(b.Shape()) {
		return gorgonia.Add(a, b)
	}

	a2, b2, err := gorgonia.Broadcast(a, b, gorgonia.NewBroadcastPattern(left, right))
	if err != nil {
		return nil, err
	}

	return gorgonia.Add(a2, b2)
}

// BroadcastHadamardProd will perform a broadcast Hadamard Prod
func BroadcastHadamardProd(a, b *gorgonia.Node, left, right []byte) (*gorgonia.Node, error) {
	if a.Shape().Eq(b.Shape()) {
		return gorgonia.HadamardProd(a, b)
	}

	a2, b2, err := gorgonia.Broadcast(a, b, gorgonia.NewBroadcastPattern(left, right))
	if err != nil {
		return nil, err
	}

	return gorgonia.HadamardProd(a2, b2)
}
