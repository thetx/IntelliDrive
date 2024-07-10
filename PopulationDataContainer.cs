using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System;
[System.Serializable]
public class PopulationDataContainer{
    
    int LAYERS = 1;
        int NEURONS = 10;
    public List<Container> list;

    public PopulationDataContainer(NNet[] population){
        list = new List<Container>();
        toSaveFun(population);
    }

    public void toSaveFun(NNet[] population){
        for(int i=0;i<population.Length;i++){
            Container c = new Container(population[i]);
            list.Add(c);
        }
    }

    public void getPopulation(NNet[] population){
        for(int a=0;a<population.Length;a++){
            NNet tempNet = new NNet();
            //biases list
            tempNet.biases = list[a].biasesList;
            //input list
            for(int i=0;i<3;i++)
                tempNet.inputLayer[0,i] = list[a].inputLayerList[i];
            //output list
            for(int i=0;i<2;i++)
                tempNet.outputLayer[0,i] = list[a].outputLayerList[i];
            //hiddenLayer list
            for(int j=0;j<LAYERS;j++){
                List<float> matrixList = list[a].hiddenLayerList[j];
                Matrix<float> f = Matrix<float>.Build.Dense(1, NEURONS);
                for(int i=0;i<NEURONS;i++){
                    // Debug.Log(population.hiddenLayers[j][0,i]+"lol");
                    f[0,i] = matrixList[i];
                }
                // Debug.Log(j);
                tempNet.hiddenLayers.Add(f);
            }
            //weights list
            for(int j=0;j<LAYERS;j++){
                float[,] matrixList = list[a].weightsList[j];
                Matrix<float> w = Matrix<float>.Build.Dense(3, NEURONS);
                for(int inp=0;inp<3;inp++){    
                    for(int i=0;i<NEURONS;i++){
                        w[inp,i] = matrixList[inp,i];
                    }
                }
                tempNet.weights.Add(w);
            }
            population[a] = tempNet;
            
        }
    }
}
[System.Serializable]
public class Container{
        int LAYERS = 1;
        int NEURONS = 10;
        public List<float> inputLayerList;
        public List<List<float>> hiddenLayerList;
        public List<float> outputLayerList;
        public List<float> biasesList;

        public List<float[,]> weightsList;
        public Container(NNet population){

            inputLayerList = new List<float>();
            outputLayerList = new List<float>();
            biasesList = new List<float>();
            hiddenLayerList = new List<List<float>>();
            weightsList = new List<float[,]>();
            saveData(population);
        }
        public void saveData(NNet population){
            biasesList = population.biases;
            for(int j=0;j<LAYERS;j++){
                List<float> matrixList = new List<float>();
                for(int i=0;i<NEURONS;i++){
                    // Debug.Log(population.hiddenLayers[j][0,i]+"lol");
                    matrixList.Add(population.hiddenLayers[j][0,i]);
                }
                // Debug.Log(j);
                hiddenLayerList.Add(matrixList);
            }
            for(int i=0;i<3;i++){
                inputLayerList.Add(population.inputLayer[0,i]);
            }
            for(int i=0;i<2;i++){
                outputLayerList.Add(population.outputLayer[0,i]);
            }
            for(int j=0;j<LAYERS;j++){
                float[,] matrixList = new float[3,NEURONS];
                for(int inp=0;inp<3;inp++){    
                    for(int i=0;i<NEURONS;i++){
                        matrixList[inp,i] = population.weights[j][inp,i];
                    }
                }
                weightsList.Add(matrixList);
            }
        }
    
    }