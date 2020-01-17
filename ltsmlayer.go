package golgi

import "gorgonia.org/gorgonia"

// LSTMLayer represents a basic LSTM layer
type LSTMLayer struct {
	inputGateWeight       gorgonia.Value
	inputGateHiddenWeight gorgonia.Value
	inputBias             gorgonia.Value

	forgetGateWeight       gorgonia.Value
	forgetGateHiddenWeight gorgonia.Value
	forgetBias             gorgonia.Value

	outputGateWeight       gorgonia.Value
	outputGateHiddenWeight gorgonia.Value
	outputBias             gorgonia.Value

	cellGateWeight       gorgonia.Value
	cellGateHiddenWeight gorgonia.Value
	cellBias             gorgonia.Value
}

func (l *LSTMLayer) getInput(g *gorgonia.ExprGraph, name string) (w whb) {
	w = newWHB(
		gorgonia.NodeFromAny(g, l.inputGateWeight, gorgonia.WithName("wix_"+name)),
		gorgonia.NodeFromAny(g, l.inputGateHiddenWeight, gorgonia.WithName("wih_"+name)),
		gorgonia.NodeFromAny(g, l.inputBias, gorgonia.WithName("bias_i_"+name)),
	)

	return
}

func (l *LSTMLayer) getForget(g *gorgonia.ExprGraph, name string) (w whb) {
	w = newWHB(
		gorgonia.NodeFromAny(g, l.forgetGateWeight, gorgonia.WithName("wfx_"+name)),
		gorgonia.NodeFromAny(g, l.forgetGateHiddenWeight, gorgonia.WithName("wfh_"+name)),
		gorgonia.NodeFromAny(g, l.forgetBias, gorgonia.WithName("bias_f_"+name)),
	)

	return
}

func (l *LSTMLayer) getOutput(g *gorgonia.ExprGraph, name string) (w whb) {
	w = newWHB(
		gorgonia.NodeFromAny(g, l.outputGateWeight, gorgonia.WithName("wox_"+name)),
		gorgonia.NodeFromAny(g, l.outputGateHiddenWeight, gorgonia.WithName("woh_"+name)),
		gorgonia.NodeFromAny(g, l.outputBias, gorgonia.WithName("bias_o_"+name)),
	)

	return
}

func (l *LSTMLayer) getCell(g *gorgonia.ExprGraph, name string) (w whb) {
	w = newWHB(
		gorgonia.NodeFromAny(g, l.cellGateWeight, gorgonia.WithName("wcx_"+name)),
		gorgonia.NodeFromAny(g, l.cellGateHiddenWeight, gorgonia.WithName("wch_"+name)),
		gorgonia.NodeFromAny(g, l.cellBias, gorgonia.WithName("bias_c_"+name)),
	)

	return
}
