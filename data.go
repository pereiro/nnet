package main

import (
	"errors"
	"fmt"
	"strconv"
)

type Marker []float64
type Payload []float64

type Data struct {
	Value int
	Marker Marker
	Payload Payload
}

func LoadFromCsv(csv [][]string,markerLen int) ([]Data,error){
	var result []Data
	for _,line := range csv{
		var d Data
		d.Marker = make([]float64,markerLen)
		d.Payload = make([]float64,len(line)-1)
		k,err := strconv.Atoi(line[0])
		if err != nil {
			return nil,err
		}
		err = d.Marker.Fill(k)
		if err != nil {
			return nil,err
		}
		d.Value = k
		for i := 1; i < len(line); i++ {
			k,err := strconv.Atoi(line[i])
			if err != nil {
				return nil,err
			}
			d.Payload[i-1] = float64(k)/255 * 0.99 + 0.01
		}
		result = append(result,d)
	}
	return result,nil
}

func (m *Marker) Fill(k int) error{
	if k<0 || k>len(*m){
		return errors.New(fmt.Sprintf("invalid marker value %d",k))
	}
	(*m)[k] = 0.99
	return nil
}