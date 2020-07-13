// +build !debug

package golgi

func enterLogScope()                            {}
func leaveLogScope()                            {}
func logf(format string, others ...interface{}) {}
func tabcount() int                             { return 0 }
