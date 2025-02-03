package br.com.qualiteti.qualitetirna.testes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.ClasseUtils;
import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronDerivativeType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;
import br.com.qualiteti.qualitetirna.rna.Acuracy;
import br.com.qualiteti.qualitetirna.rna.DataNormalizer;
import br.com.qualiteti.qualitetirna.rna.Layer;
import br.com.qualiteti.qualitetirna.rna.NeuralNet;
import br.com.qualiteti.qualitetirna.rna.definition.NetDefinition;
import br.com.qualiteti.qualitetirna.rna.neurons.Neuron;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronAdaline;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronPerceptron;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSigmoid;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSuper;
import br.com.qualiteti.qualitetirna.rna.serialization.NeuralNetSerialization;

public class TestesFase1 {
	
	public static void testePerceptron1() {
		int s;
		double previousCost;
		System.out.println("Teste 1");
		LearnResult lr = new LearnResult();
		int steps = 1001;
		List<List<Double>> x = Arrays.asList(
	            Arrays.asList(0.0, 0.0),
	            Arrays.asList(0.0, 1.0),
	            Arrays.asList(1.0, 0.0),
	            Arrays.asList(1.0, 1.0)
	        );
		List<Double> y = Arrays.asList(0.0, 0.0, 0.0, 1.0);
		Neuron neuron = new Neuron(EnNeuronType.perceptron, EnActivationType.Linear, 0.01, 0.01);
		neuron.generateWeightsAndBias();
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i), previousCost);
				previousCost = lr.costValue;
			}
			if(s%100 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
	}
	
	public static void testePerceptron1_1() {
		int s;
		double previousCost;
		System.out.println("Teste 1");
		LearnResult lr = new LearnResult();
		int steps = 1001;
		List<List<Double>> x = Arrays.asList(
	            Arrays.asList(0.0, 0.0),
	            Arrays.asList(0.0, 1.0),
	            Arrays.asList(1.0, 0.0),
	            Arrays.asList(1.0, 1.0)
	        );
		List<Double> y = Arrays.asList(0.0, 0.0, 0.0, 1.0);
		NeuronPerceptron neuron = new NeuronPerceptron();
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learnOnLine(y.get(i), previousCost);
				previousCost = lr.costValue;
			}
			if(s%100 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
	}
	
	public static void testePerceptron2() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\gruposBidimensional.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		Neuron neuron = new Neuron(EnNeuronType.perceptron, EnActivationType.Linear, 0.01, 0.01);
		
		System.out.println("Teste 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1])));
                	y.add(Double.valueOf(values[2]));
                }
            }
            //x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		Neuron n2 = new Neuron(EnNeuronType.perceptron, EnActivationType.Linear);
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.println("Predição -6,2: " + n2.predict(false) + " --Step--> " + n2.stepFunction(n2.predict(false)) );
		
	}
	public static void testePerceptron2_1() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\gruposBidimensional.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		NeuronPerceptron neuron = new NeuronPerceptron();
		
		System.out.println("Teste 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1])));
                	y.add(Double.valueOf(values[2]));
                }
            }
            //x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learnOnLine(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		NeuronPerceptron n2 = new NeuronPerceptron();
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.println("Predição -6,2: " + n2.predict() + " --Step--> " + n2.stepFunction(n2.predict()) );
		
	}
	
	public static void testePerceptron3() {  //Regressão linear Altura e peso
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\medidas.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		Neuron neuron = new Neuron(EnNeuronType.perceptron, EnActivationType.Linear, 0.01, 0.01);
		

		System.out.println("Teste 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0])));
                	y.add(Double.valueOf(values[1]));
                }
                
                //csvData.add(Arrays.asList(values));
            }
            //x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.learningRateW = 1e-7;
		neuron.learningRateB = 1e-2;
		neuron.step = false;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s,lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s,lr, neuron);
		
		
	}
	public static void testePerceptron3_1() {
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\medidas.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		NeuronSuper neuron = new NeuronPerceptron();
		

		System.out.println("Teste 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0])));
                	y.add(Double.valueOf(values[1]));
                }
                
                //csvData.add(Arrays.asList(values));
            }
            //x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.learningRateW = 1e-7;
		neuron.learningRateB = 1e-2;
		neuron.step = false;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learnOnLine(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s,lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s,lr, neuron);
		
		
	}
	
	
	
	public static void printLearnResult(int s, LearnResult lr, Neuron n) {
		System.out.print("[" + s + "] Custo:" + lr.costValue + ", bias:" + String.valueOf(n.bias).replace(".", ",") + ", Pesos:(");
		for(double w:n.weights) {
			System.out.print(String.valueOf(w).replace(".", ",") + ";");
		}
		System.out.println(")");
	}

	public static void printLearnResult(int s, LearnResult lr, NeuronSuper n) {
		System.out.print("[" + s + "] Custo:" + lr.costValue + ", bias:" + String.valueOf(n.bias).replace(".", ",") + ", Pesos:(");
		for(double w:n.weights) {
			System.out.print(String.valueOf(w).replace(".", ",") + ";");
		}
		System.out.println(")");
	}
	
	public static void printX(List<List<Double>> x) {
		String line;
		System.out.println("Valores de X:");
		for(List<Double> l:x) {
			line = "[";
			for(double v:l) {
				if(!line.equals("[")) {
					line += ", ";
				}
				line += v;
			}
			line += "]";
			System.out.println(line);
		}
		System.out.println("_".repeat(30));
	}
	
	public static void printY(List<Double> y) {
		String line;
		System.out.println("Valores de y:");
		for(double v:y) {
			line = "[" + v + "]";
			System.out.println(line);
		}
		System.out.println("_".repeat(30));
	}
	
	public static void printBiDimensional(String title, List<Double> y) {
		String line;
		System.out.println("_".repeat(30)+"\n");
		System.out.println(title + ":");
		for(double v:y) {
			line = "[" + v + "]";
			System.out.println(line);
		}
	}
	
	public static void print(String text) {
		System.out.println(text);
	}
	
	
	public static void testeAdaline1() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\gruposBidimensional.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		Neuron neuron = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear, 0.01, 0.01);
		

		System.out.println("Teste Adaline 1");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                //if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1])));
                	y.add(Double.valueOf(values[2]));
                //}
                
                //csvData.add(Arrays.asList(values));
            } 
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            //x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
            //printX(x);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		Neuron n2 = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear);
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.print("Predição ");
		for(double v:n2.entries) {
			System.out.print(v + ",");
		}
		System.out.println(": " + n2.predict(false) + " --Step--> " + n2.stepFunction(n2.predict(false)) );
		
	}
	public static void testeAdaline1_1() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\gruposBidimensional.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		NeuronSuper neuron = new NeuronAdaline();
		

		System.out.println("Teste Adaline 1");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                //if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1])));
                	y.add(Double.valueOf(values[2]));
                //}
                
                //csvData.add(Arrays.asList(values));
            } 
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            //x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
            //printX(x);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learnOnLine(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		Neuron n2 = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear);
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.print("Predição ");
		for(double v:n2.entries) {
			System.out.print(v + ",");
		}
		System.out.println(": " + n2.predict(false) + " --Step--> " + n2.stepFunction(n2.predict(false)) );
		
	}
	
	public static void testeAdaline2() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\notas.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 20001;
		int s;
		Neuron neuron = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear, 0.01, 0.01);
		

		System.out.println("Teste Adaline 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1]), Double.valueOf(values[2])));
                	y.add(Double.valueOf(values[3]));
                }
                
                //csvData.add(Arrays.asList(values));
            } 
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            x = DataNormalizer.normalizeX(x);
            //printX(x);
            //y = DataNormalizer.normalizeY(y);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		Neuron n2 = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear);
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.print("Predição ");
		for(double v:n2.entries) {
			System.out.print(v + ",");
		}
		System.out.println(": " + n2.predict(false) + " --Step--> " + n2.stepFunction(n2.predict(false)) );
		
	}
	public static void testeAdaline2_1() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\notas.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 20001;
		int s;
		NeuronSuper neuron = new NeuronAdaline();
		

		System.out.println("Teste Adaline 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1]), Double.valueOf(values[2])));
                	y.add(Double.valueOf(values[3]));
                }
                
                //csvData.add(Arrays.asList(values));
            } 
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            x = DataNormalizer.normalizeX(x);
            //printX(x);
            //y = DataNormalizer.normalizeY(y);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learnOnLine(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		NeuronSuper n2 = new NeuronAdaline();
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.print("Predição ");
		for(double v:n2.entries) {
			System.out.print(v + ",");
		}
		System.out.println(": " + n2.predict() + " --Step--> " + n2.stepFunction(n2.predict()) );
		
	}
	
	public static void testeAdaline3() {  //Porta And
		int s;
//		double absCost;
//		double targetCost = 0;
		double previousCost;
		System.out.println("Teste Adaline 3");
		LearnResult lr = new LearnResult();
		int steps = 101;
		List<List<Double>> x = Arrays.asList(
	            Arrays.asList(0.0, 0.0),
	            Arrays.asList(0.0, 1.0),
	            Arrays.asList(1.0, 0.0),
	            Arrays.asList(1.0, 1.0)
	        );
		List<Double> y = Arrays.asList(0.0, 0.0, 0.0, 1.0);
		x = DataNormalizer.normalizeX(x);
		
		Neuron neuron = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear, 0.01, 0.01);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-1);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
//			absCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i), previousCost);
				previousCost = lr.costValue;
//				absCost += Math.abs(lr.costValue);
			}
//			if(absCost <= targetCost) {	
//				break;
//			}
			if(s%10 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
	}
	
	public static void testeAdaline4() { //Grupos duas dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\gruposBidimensional.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		LearnResult lr = new LearnResult();
		double previousCost;
		int steps = 10001;
		int s;
		Neuron neuron = new Neuron(EnNeuronType.adalaine, EnActivationType.Linear);
		

		System.out.println("Teste 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]), Double.valueOf(values[1])));
                	y.add(Double.valueOf(values[2]));
                }
                
                //csvData.add(Arrays.asList(values));
            }
            x = DataNormalizer.normalizeX(x);
            //y = DataNormalizer.normalizeY(y);
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		neuron.setLearningRate(1e-2);
		neuron.step = true;
		for(s=0; s < steps; s++) {
			previousCost = 0;
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				lr = neuron.learn(y.get(i),previousCost);
				previousCost = lr.costValue;
			}
			if(s%1000 == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);
		
		Neuron n2 = new Neuron(EnNeuronType.perceptron, EnActivationType.Linear);
		n2.bias = neuron.bias;
		n2.weights = neuron.weights;
		n2.entries = Arrays.asList(-3.0,0.0);
		
		System.out.println("Predição -6,2: " + n2.predict(false) + " --Step--> " + n2.stepFunction(n2.predict(false)) );
		
	}
	
	public static void testeSigmoid1() { //Regressão logistica 1 dimensão - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\anuncios.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		List<Double> predictions = null;
		LearnResult lr = new LearnResult();
		int steps = 1001;
		int s;
		NeuronSuper neuron = new NeuronSigmoid();
		

		System.out.println("Teste Sigmoid 1");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0])));
                	y.add(Double.valueOf(values[2]));
                }
            } 
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            x = DataNormalizer.normalizeX(x);
            //printX(x);
            //y = DataNormalizer.normalizeY(y);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		
		neuron.setLearningRate(1e-3);
		neuron.step = false;
		for(s=0; s < steps; s++) {
			predictions = new ArrayList<>();
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				predictions.add(neuron.predict());
			}
//			if(s==1) {
//				printY(predictions);
//				System.out.println("b:" + neuron.bias);
//				System.out.println("w:" + neuron.weights.get(0) + " (" + neuron.weights.size() + ")");
//			}
			lr = neuron.learnOffLine(y, predictions, x);
			if(s%(steps/10) == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);	
		System.out.println("Acurácia:" + (100*Acuracy.acuracy(y, predictions, true)) + "%" );
	}
	
	public static void testeSigmoid2() { //Regressão logistica 2 dimensões - Classificação
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\anuncios.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		List<Double> predictions = null;
		LearnResult lr = new LearnResult();
		int steps = 1001;
		int s;
		NeuronSuper neuron = new NeuronSigmoid();
		

		System.out.println("Teste Sigmoid 2");
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Divide a linha usando vírgula como delimitador
                String[] values = line.split(",");
                //System.out.println(line);
                if(ClasseUtils.isNumeric(values[0])) {
                	x.add(Arrays.asList(Double.valueOf(values[0]),Double.valueOf(values[1])));
                	y.add(Double.valueOf(values[2]));
                }
            } 
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            x = DataNormalizer.normalizeX(x);
            //printX(x);
            //y = DataNormalizer.normalizeY(y);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		neuron.entries = x.get(0);
		neuron.generateWeightsAndBias();
		
		neuron.setLearningRate(1e-3);
		neuron.step = false;
		for(s=0; s < steps; s++) {
			predictions = new ArrayList<>();
			for(int i=0; i< x.size(); i++) {
				neuron.entries = x.get(i);
				predictions.add(neuron.predict());
			}
//			if(s==1) {
//				printY(predictions);
//				System.out.println("b:" + neuron.bias);
//				System.out.println("w:" + neuron.weights.get(0) + " (" + neuron.weights.size() + ")");
//			}
			lr = neuron.learnOffLine(y, predictions, x);
			if(s%(steps/10) == 0) {
				printLearnResult(s, lr, neuron);
			}
		}
		
		System.out.println("Final> Iterações:" + (s+1));
		printLearnResult(s, lr, neuron);	
		System.out.println("Acurácia:" + (100*Acuracy.acuracy(y, predictions, true)) + "%" );
	}
	
	public static void demoDerivadaSigmoid() { //comparação valores de derivada da função de ativação
		NeuronSigmoid neuron = new NeuronSigmoid();
		
		neuron.step = false;
		neuron.entries.add(-1.0);
		neuron.entries.add(-2.0);
		neuron.weights.add(2.0);
		neuron.weights.add(-3.0);
		neuron.bias = -3.0;
		List<Double> CoefEntries = neuron.predict_derivative(EnNeuronDerivativeType.entries);
		List<Double> CoefWeights = neuron.predict_derivative(EnNeuronDerivativeType.weights);
		List<Double> CoefBias = neuron.predict_derivative(EnNeuronDerivativeType.bias);
		
		System.out.println("f: " + neuron.predict());
//		System.out.println("df: " + derF);
		System.out.println ("Coeficientes:");
		System.out.println ("X0: " + CoefEntries.get(0));
		System.out.println ("X1: " + CoefEntries.get(1));
		System.out.println ("W0: " + CoefWeights.get(0));
		System.out.println ("W1: " + CoefWeights.get(1));
		System.out.println ("bias: " + CoefBias.get(0));
		
		
	}
	
	public static void rnaExercicio01() {
		NeuralNet net = new NeuralNet(Arrays.asList(0.01,0.99), ENCostFunctions.meanSquareError);
		List<Double> l1;
		List<List<Double>> l2;
		double totalError;
		List<Double> gradienteFinal;
		
		//DEFININDO A REDE  *********************************************************
		//Camada de entrada
		net.setEntryLayer(EnNeuronType.perceptron, 2, null, null);
		//Setting Values
		l2 = Arrays.asList(Arrays.asList(0.05), Arrays.asList(0.1));
		net.layers.get(0).setValues(l2);
		//setting Weights
		l2 = Arrays.asList( Arrays.asList(1.0), Arrays.asList(1.0));
		net.layers.get(0).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(0.0, 0.0);
		net.layers.get(0).setBias(l1);
		net.layers.get(0).setStepFunction(false);
		
		
		//CamadaOculta
		net.addOcultLayer(EnNeuronType.sigmoid, 2, null, null);
		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.15, 0.20), Arrays.asList(0.25,0.3));
		net.layers.get(1).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(0.35, 0.35);
		net.layers.get(1).setBias(l1);
		
		//Camada de Saída
		net.setOutLayer(EnNeuronType.sigmoid, 2, null, null);
		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.4, 0.45), Arrays.asList(0.5,0.55));
		net.layers.get(2).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(0.6, 0.6);
		net.layers.get(2).setBias(l1);
		
		net.setLearningRateForEntireNet(0.5);
		
		//FIM DA DEFINIÇÃO DA REDE **********************************************************
		
		totalError = net.doForward();
		
		print("totalError:" + totalError);
		
		printY(net.layers.get(net.layers.size()-1).forward);
		
		List<Double> gradienteCustofinal = net.getTotalCost_derivative(net.layers.get(net.layers.size()-1).getPredicts());
		
		gradienteFinal = net.doBackPropagationEntireNet();
		
		print("\nLista de gradientes:\n");
		print("Gradiente da função de custo:");
		printY(gradienteCustofinal);
		
		Layer l;
		for(int i=net.layers.size()-1; i>=0; i--) {
			l = net.layers.get(i);
			printBiDimensional("camada " + l.netOrder + ":", l.backPropagationGradient);
		}
		
		print("Gradiente Final:");
		printY(gradienteFinal);
		
		print("\nNovos Pesos:\n");
		for(int i=net.layers.size()-1; i>=0; i--) {
			l = net.layers.get(i);
			for(int n=0; n<l.neurons.size(); n++) {
				printBiDimensional("camada " + l.netOrder + " neurônio" + n + ":", l.neurons.get(n).weights);
				print("Bias: " + l.neurons.get(n).bias);
			}
		}
		
		
	}

	public static void rnaExercicio01_1() {
		NeuralNet net = new NeuralNet(Arrays.asList(0.01,0.99), ENCostFunctions.meanSquareError);
		List<Double> l1;
		List<List<Double>> l2;
		double totalError;
		List<Double> gradienteFinal;
		
		//DEFININDO A REDE  *********************************************************
		//Camada de entrada
		net.setEntryLayer(EnNeuronType.sigmoid, 2, null, null);
		//Setting Values
		l2 = Arrays.asList(Arrays.asList(0.05, 0.1), Arrays.asList(0.5, 0.1));
		net.layers.get(0).setValues(l2);
		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.15, 0.20), Arrays.asList(0.25,0.3));
		net.layers.get(0).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(0.35, 0.35);
		net.layers.get(0).setBias(l1);
		net.layers.get(0).setStepFunction(false);
		
		
//		//CamadaOculta
//		net.addOcultLayer(EnNeuronType.sigmoid, 2, null, null);
//		//setting Weights
//		l2 = Arrays.asList( Arrays.asList(0.15, 0.20), Arrays.asList(0.25,0.3));
//		net.layers.get(1).setWeight(l2);
//		//Setting Bias
//		l1 = Arrays.asList(0.35, 0.35);
//		net.layers.get(1).setBias(l1);
		
		//Camada de Saída
		net.setOutLayer(EnNeuronType.sigmoid, 2, null, null);
		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.4, 0.45), Arrays.asList(0.5,0.55));
		net.layers.get(1).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(0.6, 0.6);
		net.layers.get(1).setBias(l1);
		
		net.setLearningRateForEntireNet(0.5);
		
		//FIM DA DEFINIÇÃO DA REDE **********************************************************
		
		totalError = net.doForward();
		
		print("totalError:" + totalError);
		
		printY(net.layers.get(net.layers.size()-1).forward);
		
		List<Double> gradienteCustofinal = net.getTotalCost_derivative(net.layers.get(net.layers.size()-1).getPredicts());
		
		gradienteFinal = net.doBackPropagationEntireNet();
		
		print("\nLista de gradientes:\n");
		print("Gradiente da função de custo:");
		printY(gradienteCustofinal);
		
		Layer l;
		for(int i=net.layers.size()-1; i>=0; i--) {
			l = net.layers.get(i);
			printBiDimensional("camada " + l.netOrder + ":", l.backPropagationGradient);
		}
		
		print("Gradiente Final:");
		printY(gradienteFinal);
		
		print("\nNovos Pesos:\n");
		for(int i=net.layers.size()-1; i>=0; i--) {
			l = net.layers.get(i);
			for(int n=0; n<l.neurons.size(); n++) {
				printBiDimensional("camada " + l.netOrder + " neurônio" + n + ":", l.neurons.get(n).weights);
				print("Bias: " + l.neurons.get(n).bias);
			}
		}
		
		
	}

	public static void rnaExercicio2() {
		NeuralNet net = new NeuralNet(Arrays.asList(1.0, 0.0, 0.0), ENCostFunctions.negLogLikelihood);
		List<Double> l1;
		List<List<Double>> l2;
		double totalError;
		
		//Camada de Entrada
		net.setEntryLayer(EnNeuronType.flex, 3, EnActivationType.ReLU, ENCostFunctions.crossEntropy);
		//Setting Values
		l2 = Arrays.asList(Arrays.asList(0.1, 0.2, 0.7), Arrays.asList(0.1, 0.2, 0.7), Arrays.asList(0.1, 0.2, 0.7));
		net.layers.get(0).setValues(l2);
		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.1, 0.2, 0.3), Arrays.asList(0.3, 0.2, 0.7), Arrays.asList(0.4,0.3, 0.9));
		net.layers.get(0).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(1.0, 1.0, 1.0);
		net.layers.get(0).setBias(l1);
		net.layers.get(0).setStepFunction(false);
		
		//Camada h1
		net.addOcultLayer(EnNeuronType.sigmoid, 3, null, null);

		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.2, 0.3, 0.5), Arrays.asList(0.3, 0.5, 0.7), Arrays.asList(0.6,0.4, 0.8));
		net.layers.get(1).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(1.0, 1.0, 1.0);
		net.layers.get(1).setBias(l1);
		net.layers.get(1).setStepFunction(false);
		
		
		//Camada out
		net.setOutLayer(EnNeuronType.perceptron, 3, null, null);

		//setting Weights
		l2 = Arrays.asList( Arrays.asList(0.1, 0.4, 0.8), Arrays.asList(0.3, 0.7, 0.2), Arrays.asList(0.5,0.2, 0.9));
		net.layers.get(2).setWeight(l2);
		//Setting Bias
		l1 = Arrays.asList(1.0, 1.0, 1.0);
		net.layers.get(2).setBias(l1);
		net.layers.get(2).setStepFunction(false);
		
		//Configurações finais da net
		net.setLearningRateForEntireNet(0.1);
		
		//*******************************************************************
		//                     LOOP
		//*******************************************************************
		
		int steps = 301;
		int printsInterval = 30;
		String line;
		String line2;
		// List<Double> gradienteNet;
		for(int s=0; s<steps; s++) {
			totalError =  net.doForward();
			//gradienteNet = 
			net.doBackPropagationEntireNet();
			if(s%printsInterval == 0) {
				print("Total Error:" + totalError);
				//print("GradienteNet: " + gradienteNet);
			}
		}
		
		print("Parâmetros:");
		for(Layer l:net.layers) {
			line = "";
			line2 = "";
			for(NeuronSuper n:l.neurons) {
				line += "[";
				line2 += "[";
				for(double w:n.weights) {
					line += w +" ";
				}
				line += "]";
				line2 += n.bias + "]";
			}
			print("Pesos_Camada" + l.netOrder + ":" + line  );
			print("Bias_Camada" + l.netOrder + ":" + line2  );
		}
		
		print("RESULTADO");
		print(">" + net.getNetPredict(Arrays.asList(0.9, 0.2, 0.07), true));
		print("");
		try {
			print(Base64.getEncoder().encodeToString(  NeuralNetSerialization.serialize(net.export())  ));
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		
	}
	
	public static void testeRecuperacao() {
		String stNet = "rO0ABXNyADpici5jb20ucXVhbGl0ZXRpLnF1YWxpdGV0aXJuYS5ybmEuZGVmaW5pdGlvbi5OZXREZWZpbml0aW9uAAAAAAAAAAECAAJMAAxjb3N0RnVuY3Rpb250ADxMYnIvY29tL3F1YWxpdGV0aS9xdWFsaXRldGlybmEvY29tbW9uL2VudW1zL0VOQ29zdEZ1bmN0aW9ucztMABFsYXllcnNEZWZpbml0aW9uc3QAEExqYXZhL3V0aWwvTGlzdDt4cH5yADpici5jb20ucXVhbGl0ZXRpLnF1YWxpdGV0aXJuYS5jb21tb24uZW51bXMuRU5Db3N0RnVuY3Rpb25zAAAAAAAAAAASAAB4cgAOamF2YS5sYW5nLkVudW0AAAAAAAAAABIAAHhwdAAQbmVnTG9nTGlrZWxpaG9vZHNyABNqYXZhLnV0aWwuQXJyYXlMaXN0eIHSHZnHYZ0DAAFJAARzaXpleHAAAAADdwQAAAADc3IAPGJyLmNvbS5xdWFsaXRldGkucXVhbGl0ZXRpcm5hLnJuYS5kZWZpbml0aW9uLkxheWVyRGVmaW5pdGlvbgAAAAAAAAABAgAFRAARbGF5ZXJMZWFybmluZ1JhdGVJAAhuZXRPcmRlckwACWxheWVyVHlwZXQAOExici9jb20vcXVhbGl0ZXRpL3F1YWxpdGV0aXJuYS9jb21tb24vZW51bXMvRW5MYXllclR5cGU7TAASbmV1cm9uc0RlZmluaXRpb25zcQB+AAJMAAtuZXVyb25zVHlwZXQAOUxici9jb20vcXVhbGl0ZXRpL3F1YWxpdGV0aXJuYS9jb21tb24vZW51bXMvRW5OZXVyb25UeXBlO3hwQFkAAAAAAAAAAAAAfnIANmJyLmNvbS5xdWFsaXRldGkucXVhbGl0ZXRpcm5hLmNvbW1vbi5lbnVtcy5FbkxheWVyVHlwZQAAAAAAAAAAEgAAeHEAfgAFdAAFaW5wdXRzcQB+AAgAAAADdwQAAAADc3IAPWJyLmNvbS5xdWFsaXRldGkucXVhbGl0ZXRpcm5hLnJuYS5kZWZpbml0aW9uLk5ldXJvbkRlZmluaXRpb24AAAAAAAAAAQIACUQACGFsZmFSZUxVRAAEYmlhc0QADWxlYXJuaW5nUmF0ZUJEAA1sZWFybmluZ1JhdGVXWgAEc3RlcEwAEmFj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";
		byte[] vet = Base64.getDecoder().decode(stNet);
		NetDefinition def = null;
		try {
			def = NeuralNetSerialization.deserialize(vet);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		NeuralNet net = new NeuralNet(def, Arrays.asList(0.1, 0.2, 0.7));
		print("RESULTADO");
		print(">" + net.getNetPredict(true));
		
		
	}
	
}
