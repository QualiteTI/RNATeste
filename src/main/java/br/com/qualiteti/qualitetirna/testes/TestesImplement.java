package br.com.qualiteti.qualitetirna.testes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.ClasseUtils;
import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;
import br.com.qualiteti.qualitetirna.rna.DataNormalizer;
import br.com.qualiteti.qualitetirna.rna.NeuralNet;
import br.com.qualiteti.qualitetirna.rna.definition.NetDefinition;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSigmoid;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSuper;

public class TestesImplement {

	
	
	public static void testeSigmoidFase1() {
		String filePath = "C:\\Cursos\\Manual-Pratico-Deep-Learning-master\\data\\anuncios.csv";
		List<List<Double>> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		List<Double> predictions = null;
		LearnResult lr = new LearnResult();
		int steps = 500;
		NeuralNet net;
		
		//LEITURA DE DADOS
		System.out.println("Teste Sigmoid 1");
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
            System.out.println("Dados Carregados:" + x.size() + "/" + y.size());
            x = DataNormalizer.normalizeX(x);
            //printX(x);
            //y = DataNormalizer.normalizeY(y);

            System.out.println("Dados Normalizados:" + x.size() + "/" + y.size());
        } catch (IOException e) {
            System.err.println("Erro ao ler o arquivo CSV: " + e.getMessage());
        }
		
		
		//DEFINIÇÃO DA RNA
		net = new NeuralNet(ENCostFunctions.crossEntropy );
		net.setEntryLayer(EnNeuronType.perceptron, 1, null, null);
		for(int i=0; i<5;i++) {
			net.addOcultLayer(EnNeuronType.flex, 20, EnActivationType.LeakyReLU, ENCostFunctions.crossEntropy);
		}
		net.setOutLayer(EnNeuronType.sigmoid, 1, null, null);
		net.costFunction = ENCostFunctions.sigmoidCrossEntropy;
		net.setLearningRateForEntireNet(0.005);
		net.generateWeightsAndBias(x.get(0).size());  //Definie a quantidade e aleatoriza os pesos e bias de todos os neurônios da rede
		//net.layers.get(net.layers.size()-1).setStepFunction(true);
		NetDefinition def = net.export();
		
		
		//LOOP
		int sampleIni = 0;
		int sampleEnd = 0;
		int subSampleSize;
		List<List<Double>> subSample;
		List<Double> subY;
		double totalCost = Double.MAX_VALUE;
		for(int s=0;s<steps; s++) {
			
			
			for(int i=0; i<x.size(); i++) {
				subSample = new ArrayList<>();
				subSample.add(x.get(i));
				subY = Arrays.asList(y.get(i));
				net.realValues = subY;
				totalCost = net.doForward(subSample);
				net.doBackPropagationEntireNet();
			}
//			sampleIni = 0;
//			sampleEnd = 9;
//			subSampleSize = sampleEnd - sampleIni + 1;
//			
//			while(sampleEnd < x.size()-1) {
//				subSample = new ArrayList<>();
//				subY = new ArrayList<>();
//				for(int i=sampleIni; i<= sampleEnd; i++) {
//					subSample.add(x.get(i));
//					subY.add(y.get(i));
//				}
//				net.realValues = subY;
//				totalCost = net.doForward(subSample);
//				net.doBackPropagationEntireNet();
//				sampleIni += subSampleSize;
//				sampleEnd += subSampleSize;
//			}
			if(s%(Math.max(steps/10, 1))==0) {
				print("Step[" + s + "]TotalCost:" + totalCost);
			}
		}
		
		print("TotalCost:" + totalCost);
		
	}
	
	
	//IMPRESSÃO
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
	
}
