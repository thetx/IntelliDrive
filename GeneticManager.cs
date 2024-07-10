using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;

public class GeneticManager : MonoBehaviour
{
    [Header("References")]
    public CarController controller;
    public GameObject carPrefab;
    
    Dictionary<GameObject, CarController> carControllerArray = new Dictionary<GameObject, CarController>();
    Dictionary<GameObject, int> carControllerIndexArray = new Dictionary<GameObject, int>();

    [Header("Controls")]
    public int initialPopulation = 85;
    [Range(0.0f, 1.0f)]
    public float mutationRate = 0.055f;

    [Header("Crossover Controls")]
    public int bestAgentSelection = 8;
    public int worstAgentSelection = 3;
    public int numberToCrossover;

    private List<int> genePool = new List<int>();

    private int naturallySelected;

    public NNet[] population;

    [Header("Public View")]
    public int currentGeneration;
    public int currentGenome = 0;
    public int deaths = 0;

    private void Start()
    {
        CreatePopulation();
    }

    private void CreatePopulation()
    {
        population = new NNet[initialPopulation];
        FillPopulationWithRandomValues(population, 0);
        for(int i=0;i<initialPopulation;i++){
            GameObject car = Instantiate(carPrefab);
            carControllerIndexArray.Add(car,i);
            carControllerArray.Add(car,car.GetComponent<CarController>());
        }
        ResetToCurrentGenome();
    }

    private void ResetToCurrentGenome()
    {
        foreach(var pair in carControllerArray){
            CarController carController = pair.Value;
            carController.ResetWithNetwork(population[carControllerIndexArray[pair.Key]]);
            GameObject car = pair.Key;
            car.SetActive(true);
        }
        // controller.ResetWithNetwork(population[currentGenome]);
    }

    private void FillPopulationWithRandomValues (NNet[] newPopulation, int startingIndex)
    {
        while (startingIndex < initialPopulation)
        {
            newPopulation[startingIndex] = new NNet();
            newPopulation[startingIndex].Initialise(controller.LAYERS, controller.NEURONS);
            startingIndex++;
        }
    }

    
    public void Death (float fitness, NNet network,GameObject car)
    {
        if (currentGenome < population.Length -1)
        {
            population[carControllerIndexArray[car]].fitness = fitness;
            // population[currentGenome].fitness = fitness;
            currentGenome++;
            // ResetToCurrentGenome();
            car.SetActive(false);
        }
        else
        {
            RePopulate();
        }

    }

    
    private void RePopulate()
    {
        SaveToJson();
        if(currentGeneration == 2){
            // loadFromJson();
        }

        genePool.Clear();
        currentGeneration++;
        naturallySelected = 0;
        SortPopulation();

        NNet[] newPopulation = PickBestPopulation();

        Crossover(newPopulation);
        Mutate(newPopulation);

        FillPopulationWithRandomValues(newPopulation, naturallySelected);

        population = newPopulation;

        currentGenome = 0;
        deaths=0;

        ResetToCurrentGenome();

    }

    private void Mutate (NNet[] newPopulation)
    {

        for (int i = 0; i < naturallySelected; i++)
        {

            for (int c = 0; c < newPopulation[i].weights.Count; c++)
            {

                if (Random.Range(0.0f, 1.0f) < mutationRate)
                {
                    newPopulation[i].weights[c] = MutateMatrix(newPopulation[i].weights[c]);
                }

            }

        }

    }

    Matrix<float> MutateMatrix (Matrix<float> A)
    {

        int randomPoints = Random.Range(1, (A.RowCount * A.ColumnCount) / 7);

        Matrix<float> C = A;

        for (int i = 0; i < randomPoints; i++)
        {
            int randomColumn = Random.Range(0, C.ColumnCount);
            int randomRow = Random.Range(0, C.RowCount);

            C[randomRow, randomColumn] = Mathf.Clamp(C[randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
        }

        return C;

    }

    private void Crossover (NNet[] newPopulation)
    {
        for (int i = 0; i < numberToCrossover; i+=2)
        {
            int AIndex = i;
            int BIndex = i + 1;

            if (genePool.Count >= 1)
            {
                for (int l = 0; l < 100; l++)
                {
                    AIndex = genePool[Random.Range(0, genePool.Count)];
                    BIndex = genePool[Random.Range(0, genePool.Count)];

                    if (AIndex != BIndex)
                        break;
                }
            }

            NNet Child1 = new NNet();
            NNet Child2 = new NNet();

            Child1.Initialise(controller.LAYERS, controller.NEURONS);
            Child2.Initialise(controller.LAYERS, controller.NEURONS);

            Child1.fitness = 0;
            Child2.fitness = 0;


            for (int w = 0; w < Child1.weights.Count; w++)
            {

                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    Child1.weights[w] = population[AIndex].weights[w];
                    Child2.weights[w] = population[BIndex].weights[w];
                }
                else
                {
                    Child2.weights[w] = population[AIndex].weights[w];
                    Child1.weights[w] = population[BIndex].weights[w];
                }

            }


            for (int w = 0; w < Child1.biases.Count; w++)
            {

                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    Child1.biases[w] = population[AIndex].biases[w];
                    Child2.biases[w] = population[BIndex].biases[w];
                }
                else
                {
                    Child2.biases[w] = population[AIndex].biases[w];
                    Child1.biases[w] = population[BIndex].biases[w];
                }

            }

            newPopulation[naturallySelected] = Child1;
            naturallySelected++;

            newPopulation[naturallySelected] = Child2;
            naturallySelected++;

        }
    }

    private NNet[] PickBestPopulation()
    {

        NNet[] newPopulation = new NNet[initialPopulation];

        for (int i = 0; i < bestAgentSelection; i++)
        {
            newPopulation[naturallySelected] = population[i].InitialiseCopy(controller.LAYERS, controller.NEURONS);
            newPopulation[naturallySelected].fitness = 0;
            naturallySelected++;
            
            int f = Mathf.RoundToInt(population[i].fitness * 10);

            for (int c = 0; c < f; c++)
            {
                genePool.Add(i);
            }

        }

        for (int i = 0; i < worstAgentSelection; i++)
        {
            int last = population.Length - 1;
            last -= i;

            int f = Mathf.RoundToInt(population[last].fitness * 10);

            for (int c = 0; c < f; c++)
            {
                genePool.Add(last);
            }

        }

        return newPopulation;

    }

    private void SortPopulation()
    {
        for (int i = 0; i < population.Length; i++)
        {
            for (int j = i; j < population.Length; j++)
            {
                if (population[i].fitness < population[j].fitness)
                {
                    NNet temp = population[i];
                    population[i] = population[j];
                    population[j] = temp;
                }
            }
        }

    }

    public void SaveToJson(){
        PopulationDataContainer pd = new PopulationDataContainer(population);
        string populationToSave = JsonUtility.ToJson(pd);
        string filePath = Application.persistentDataPath + "/PopulationData.json";
        // Debug.Log(toSave);
        System.IO.File.WriteAllText(filePath, populationToSave);
        // Debug.Log("Done");
    }

    // public void loadFromJson(){
    //     string filePath = Application.persistentDataPath + "/PopulationData.json";
    //     string saveData = System.IO.File.ReadAllText(filePath);
    //     PopulationDataContainer pd = JsonUtility.FromJson<PopulationDataContainer>(saveData);
    //     // pd.getPopulation(population);

    //     // Debug.Log(population[1].biases[2]);
    //     // Debug.Log(population[1].hiddenLayers[0][0,2]);
    //     // Debug.Log(population[1].weights[0][1,4]);
    //     // ResetToCurrentGenome();
    // }

}
