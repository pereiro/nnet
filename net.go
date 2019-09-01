package main

import (
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math"
	"math/rand"
)

type NNet struct {
	InputNodes      int
	HiddenNodes     int
	OutputNodes     int
	LearningRate    float64
	WInputHidden    *mat.Dense
	WHiddenOutput   *mat.Dense
	RawInputHeader  []byte
	RawHiddenOutput []byte
}

func (n *NNet) Save(filename string) error{
	var err error
	n.RawInputHeader,err = n.WInputHidden.MarshalBinary()
	if err != nil {
		return err
	}
	n.RawHiddenOutput,err = n.WHiddenOutput.MarshalBinary()
	if err != nil {
		return err
	}
	data,err := json.Marshal(n)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename,data,0777)
}

func (n *NNet) Load(filename string) error {
	data,err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	err = json.Unmarshal(data,n)
	if err != nil {
		return err
	}
	n.WHiddenOutput.Reset()
	n.WInputHidden.Reset()
	err = n.WHiddenOutput.UnmarshalBinary(n.RawHiddenOutput)
	if err != nil {
		return err
	}
	err = n.WInputHidden.UnmarshalBinary(n.RawInputHeader)
	if err != nil {
		return err
	}
	return nil
}

func sigmoid(x float64) float64{
	return 1.0 / (1.0 + math.Exp(-x))
}

func (n *NNet) Query(inputs *mat.Dense) *mat.Dense{
	hiddenInputs := new(mat.Dense)
	hiddenInputs.Mul(n.WInputHidden,inputs)
	hiddenOutputs := new(mat.Dense)
	hiddenOutputs.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, hiddenInputs)
    finalInputs := new(mat.Dense)
    finalInputs.Mul(n.WHiddenOutput,hiddenOutputs)
    finalOutputs := new(mat.Dense)
    finalOutputs.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, finalInputs)
    return finalOutputs
}

func (n *NNet) Train(inputs *mat.Dense,targets *mat.Dense) {

	hiddenInputs := new(mat.Dense)
	hiddenInputs.Mul(n.WInputHidden,inputs)
	hiddenOutputs := new(mat.Dense)
	hiddenOutputs.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, hiddenInputs)
	finalInputs := new(mat.Dense)
 	finalInputs.Mul(n.WHiddenOutput,hiddenOutputs)
	finalOutputs := new(mat.Dense)
	finalOutputs.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, finalInputs)
	outputErrors := new(mat.Dense)
	outputErrors.Sub(targets,finalOutputs)
	hiddenErrors := new(mat.Dense)
	hiddenErrors.Mul(n.WHiddenOutput.T(),outputErrors)

    n.WHiddenOutput.Add(n.WHiddenOutput,getDelta(outputErrors,finalOutputs,hiddenOutputs,n.LearningRate))
	n.WInputHidden.Add(n.WInputHidden,getDelta(hiddenErrors,hiddenOutputs,inputs,n.LearningRate))
}

func getDelta(errors *mat.Dense, lastOutputs *mat.Dense, prevOutputs *mat.Dense,lr float64) *mat.Dense{
	delta := new(mat.Dense)
	outputMul := new(mat.Dense)
	outputMul.MulElem(errors, lastOutputs)
	lastOutputs.Apply(func(_, _ int, v float64) float64 {
		return 1-v
	}, lastOutputs)
	outputMul.MulElem(outputMul, lastOutputs)
	delta.Mul(outputMul, prevOutputs.T())
	delta.Apply(func(_, _ int, v float64) float64 {
		return v*lr
	},delta)
	return delta
}


func NewNNet(inodes int,hnodes int,onodes int,lr float64) *NNet{
	rand.Seed(0)
	n := NNet{
		InputNodes:   inodes,
		HiddenNodes:  hnodes,
		OutputNodes:  onodes,
		LearningRate: lr,
	}
	wihValues := make([]float64,inodes*hnodes)
	wohValues := make([]float64,onodes*hnodes)
	setInitialWeights(wihValues)
	setInitialWeights(wohValues)
	n.WInputHidden = mat.NewDense(hnodes,inodes,wihValues)
	n.WHiddenOutput = mat.NewDense(onodes,hnodes,wohValues)
	return &n
}

func setInitialWeights(arr []float64){
	for i := 0; i < len(arr); i++ {
		arr[i] = rand.Float64()-0.5
	}
}

func mPrint(name string,m *mat.Dense){
	r,c := m.Dims()
	fmt.Printf("%s %dx%d\n",name,r,c)
}
