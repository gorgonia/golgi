package golgi

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
)

func RMS(yHat, y *G.Node) (retVal *G.Node, err error) {
	if retVal, err = G.Sub(yHat, y); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)")
	}
	if retVal, err = G.Square(retVal); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)²")
	}
	if retVal, err = G.Mean(retVal); err != nil {
		return nil, errors.Wrap(err, "mean((ŷ-y)²)")
	}
	return
}
