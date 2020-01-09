package golgi

import (
	"log"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// trace is a Layer used for debugging
type trace struct {
	name              string
	format, errFormat string
	logger            *log.Logger
}

// Trace creates a layer for debugging composed layers
//
// The format string adds four things: "%s %v (%p) %v" - name (of trace), x, x, x.Shape()
func Trace(name, format, errFormat string, logger *log.Logger) *trace {
	const (
		def    = "\t%s %v (%p) %v"
		defErr = "ERR %s %v"
	)
	if format == "" {
		format = def
	}
	if errFormat == "" {
		errFormat = defErr
	}
	return &trace{
		name:      name,
		format:    format,
		errFormat: errFormat,
		logger:    logger,
	}
}

func (t *trace) Model() G.Nodes { return nil }
func (t *trace) Fwd(x G.Input) G.Result {
	err := G.CheckOne(x)
	var print func(string, ...interface{})

	print = log.Printf
	if t.logger != nil {
		print = t.logger.Printf
	}

	if err != nil {
		print(t.errFormat, err)
		return G.LiftResult(x, nil)
	}
	print(t.format, t.name, x, x, x.Node().Shape())
	return G.LiftResult(x, nil)
}
func (t *trace) Name() string        { return t.name }
func (t *trace) Type() hm.Type       { return nil }
func (t *trace) Shape() tensor.Shape { return nil }
func (t *trace) Describe()           {}
