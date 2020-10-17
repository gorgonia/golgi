package golgi

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func wrapErr(t Term, template string, args ...interface{}) gorgonia.Result {
	name := t.Name()
	if name == "" {
		name = "<unnamed>"
	}

	args = append([]interface{}{t, name}, args...)

	return gorgonia.Err(
		fmt.Errorf("[%T::%s] "+template, args...),
	)
}
