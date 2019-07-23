package golgi

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
)

type maebe struct {
	err error
}

// generic monad... may be useful
func (m *maebe) do(f func() (*G.Node, error)) (retVal *G.Node) {
	if m.err != nil {
		return nil
	}
	if retVal, m.err = f(); m.err != nil {
		m.err = errors.WithStack(m.err)
	}
	return
}

type Err struct{ E error }

func (err Err) Node() *G.Node  { return nil }
func (err Err) Nodes() G.Nodes { return nil }
func (err Err) Err() error     { return err.E }

func lift(a *G.Node, err error) Result {
	if err != nil {
		return Err{err}
	}
	return a
}

// CheckOne checks whether an input is an error
func CheckOne(in Input) error {
	if errer, ok := in.(Errer); ok && errer.Err() != nil {
		return errer.Err()
	}
	return nil
}
