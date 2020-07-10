package golgi

import (
	"fmt"

	//. "gorgonia.org/golgi"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/qol"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

func ExampleEmbedding_1() {
	const (
		N        = 13
		dims     = 50
		sentence = 10
	)
	g := G.NewGraph()
	x := G.NewVector(g, qol.ClassType(), G.WithShape(sentence), G.WithInit(G.Zeroes()), G.WithName("sentence")) // WithInit or WithValue is required.
	w := G.NewMatrix(g, tensor.Float64, G.WithShape(N, dims), G.WithName("embW"), G.WithInit(G.GlorotN(1)))
	emb1 := NewEmbedding(WithWeights(w), WithName("emb1"), WithClasses(N), WithBatchSize(sentence))
	emb2 := NewEmbedding(Of(tensor.Float64), WithSize(dims), WithClasses(N), WithBatchSize(sentence))
	sel1 := emb1.Fwd(x)
	sel2 := emb2.Fwd(x)

	classes := x.Value().Data().([]qol.Class)
	classes[1] = 1

	// Run() is required to be run everytime the VM is run.
	// This is because the embedding layer relies on mechanisms external to the graphs' system.
	if err := emb1.Run(x); err != nil {
		fmt.Println(err)
	}
	if err := emb2.Run(x); err != nil {
		fmt.Println(err)
	}
	vm := G.NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Println(err)
	}
	fmt.Printf("sel1: %v\nsel2: %v\n", sel1.Node().Shape(), sel2.Node().Shape())

	mat1, err := native.Matrix(sel1.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat2, err := native.Matrix(sel2.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	// only the 2nd word of the sentence doesn't have word ID 0
	testFn := func(mat [][]float64, name string) {
		for i := range mat {
			if i == 1 {
				continue
			}
			var eq int
			for j := range mat[i] {
				if mat[i][j] == mat[1][j] { // strictly speaking we should use a better float cmp
					eq++
				}
			}
			if eq >= len(mat[i])/2 {
				fmt.Printf("%s Row %d is the same as Row 1.\n%1.1f\n=\n%1.1f\n", name, i, mat[i], mat[1])
			}
		}
	}
	testFn(mat1.([][]float64), "mat1")
	testFn(mat2.([][]float64), "mat2")

	// Output:
	// sel1: (10, 50)
	// sel2: (10, 50)

}

func ExampleEmbedding_2() {
	const (
		N         = 13
		dims      = 50
		sentence  = 10
		batchSize = 10
	)
	g := G.NewGraph()
	x := G.NewMatrix(g, qol.ClassType(), G.WithShape(batchSize, sentence), G.WithInit(G.Zeroes()), G.WithName("sentence")) // WithInit or WithValue is required.
	w := G.NewMatrix(g, tensor.Float64, G.WithShape(N, dims), G.WithName("embW"), G.WithInit(G.GlorotN(1)))
	emb1 := NewEmbedding(WithWeights(w), WithName("emb1"), WithSize(dims), WithClasses(N), WithBatchSize(batchSize*sentence))
	emb2 := NewEmbedding(Of(tensor.Float64), WithSize(dims), WithClasses(N), WithBatchSize(batchSize*sentence))

	sel1 := emb1.Fwd(x)
	sel2 := emb2.Fwd(x)

	classes := x.Value().Data().([]qol.Class)
	classes[1] = 1

	// Run() is required to be run everytime the VM is run.
	// This is because the embedding layer relies on mechanisms external to the graphs' system.
	if err := emb1.Run(x); err != nil {
		fmt.Println(err)
	}
	if err := emb2.Run(x); err != nil {
		fmt.Println(err)
	}
	vm := G.NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("sel1: %v\nsel2: %v\n", sel1.Node().Shape(), sel2.Node().Shape())

	mat1, err := native.Tensor3F64(sel1.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat2, err := native.Tensor3F64(sel2.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}
	// only the 2nd word of the sentence doesn't have word ID 0
	testFn := func(mat [][][]float64, name string) {
		for i := range mat {
			for j := range mat[i] {
				if i == 0 && j == 1 {
					continue
				}
				var eq int
				for k := range mat[i][j] {
					if mat[i][j][k] == mat[0][1][k] { // strictly speaking we should use a better float cmp
						eq++
					}
				}
				if eq >= len(mat[i][j])/2 {
					fmt.Printf("%s Row %d is the same as Row 1.\n%1.1f\n=\n%1.1f\n", name, j, mat[i][j], mat[0][1])
				}
			}

		}
	}
	testFn(mat1, "mat1")
	testFn(mat2, "mat2")

	// Output:
	// sel1: (10, 10, 50)
	// sel2: (10, 10, 50)

}
