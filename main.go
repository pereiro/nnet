package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
)

var(
	iNodes       = flag.Int("inodes",784,"input nodes")
	hNodes       = flag.Int("hnodes",200,"hidden nodes")
	oNodes       = flag.Int("onodes",10,"output nodes")
	learningRate = flag.Float64("lr",0.2,"learning rate")
	trainDataPath = flag.String("traindata","data/mnist_train.csv","path to train data")
	testDataPath = flag.String("testdata","data/mnist_test.csv","path to test data")
	modelPath  = flag.String("path to model file","test.net","path to trained model")
	saveModel = flag.Bool("save",false,"save model to file")
	loadModel = flag.Bool("load",false,"load model from file")
	trainModel = flag.Bool("train",false,"train model")
	trainEpochs = flag.Int("epochs",1,"training epochs count")
	testModel = flag.Bool("test",false,"test model")
)

func main() {
	flag.Parse()
	n := NewNNet(*iNodes,*hNodes,*oNodes,*learningRate)

	if *loadModel{
		err := n.Load(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
	}

	if *trainModel {
		for epoch := 1; epoch<=*trainEpochs; epoch++ {
			log.Printf("Epoch %d training started\n",epoch)
			trainPath, _ := os.Open(*trainDataPath)
			trainFile := csv.NewReader(bufio.NewReader(trainPath))

			trainCSV, err := trainFile.ReadAll()
			if err != nil {
				log.Fatal(err)
			}

			trainData, err := LoadFromCsv(trainCSV, *oNodes)
			if err != nil {
				log.Fatal(err)
			}
			for _, t := range trainData {
				n.Train(mat.NewDense(*iNodes, 1, t.Payload), mat.NewDense(*oNodes, 1, t.Marker))
			}
			log.Printf("Epoch %d training completed\n",epoch)

			if *testModel{
				log.Printf("Epoch %d testing started\n",epoch)
				testPath,_ := os.Open(*testDataPath)
				testFile := csv.NewReader(bufio.NewReader(testPath))

				testCSV, err := testFile.ReadAll()
				if err != nil {
					log.Fatal(err)
				}

				testData,err := LoadFromCsv(testCSV,*oNodes)
				if err != nil {
					log.Fatal(err)
				}

				successNum := 0.0

				for _,t := range testData{
					result := n.Query(mat.NewDense(*iNodes,1,t.Payload))
					var maxVal float64
					maxIndex := 0
					result.Apply(func(i, _ int, v float64) float64 {
						if v > maxVal {
							maxVal = v
							maxIndex = i
						}
						return v
					},result)
					if maxIndex == t.Value {
						successNum++
					}
					//fa := mat.Formatted(result.T(),mat.Prefix("    "), mat.Squeeze())
					//fmt.Printf("%t || res = %v || predicted = %.2g\n\n",success,t.Marker,fa)
				}
				log.Printf("Epoch %d testing completed, result =  %f \n",epoch,successNum/float64(len(testData))*100)
			}

		}

	}

	if *saveModel{
		err := n.Save(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
	}

}

