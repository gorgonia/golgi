package golgi

import "gorgonia.org/gorgonia"

func newWHB(wx, wh, b *gorgonia.Node) (out whb) {
	out.wx = wx
	out.wh = wh
	out.b = b
	return
}

type whb struct {
	wx *gorgonia.Node
	wh *gorgonia.Node
	b  *gorgonia.Node
}
