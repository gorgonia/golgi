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

type aritier interface {
	arity() int
}

type l1 func(a *G.Node) (*G.Node, error)
type l2 func(a, b *G.Node) (*G.Node, error)

func (l1) arity() int { return 1 }
func (l2) arity() int { return 2 }

func lift(f aritier, inputs ...*G.Node) func() (*G.Node, error) {
	return func() (*G.Node, error) {
		ar := f.arity()
		if len(inputs) != ar {
			return nil, errors.Errorf("Expected %d inputs. Got %d instead", ar, len(inputs))
		}
		switch ar {
		case 1:
			fn := f.(l1)
			return fn(inputs[0])
		case 2:
			fn := f.(l2)
			return fn(inputs[0], inputs[1])
		default:
			return nil, errors.New("Unhandled arity")
		}
	}
}
