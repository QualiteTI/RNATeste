package br.com.qualiteti.qualitetirna.rna.neurons;

import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;

public class Neuron {
	public EnNeuronType type;
	public EnActivationType activation;
	public double bias = 0;
	public double alfaReLU = 0.1;
	public double learningRateW = 0.1;
	public double learningRateB = 0.1;
	public boolean step = false;
	public List<Double> entries = new ArrayList<>();
	public List<Double> weights = new ArrayList<>();
	
	public Neuron(EnNeuronType type, EnActivationType activation) {
		this.type = type;
		this.activation = activation;
	}
	
	public Neuron(EnNeuronType type, EnActivationType activation, double learningRateW, double learningRateB) {
		this.type = type;
		this.activation = activation;
		this.learningRateW = learningRateW;
		this.learningRateB = learningRateB;
	}
	
	public void setLearningRate(double learningRate) {
		this.learningRateW = learningRate;
		this.learningRateB = learningRate;
	}
	
	public void generateWeightsAndBias() {
		weights = new ArrayList<>();
		for(int i=0; i<this.entries.size();i++) {
			this.weights.add(2*Math.random()-1);
		}
		this.bias = 2*Math.random()-1;
	}
	

	
	//Funções de Aprendizagem
	public LearnResult learn(double realResult, double previousCost) {
		double prediction = predict(false);
		LearnResult out = new LearnResult();
		//out.realValue = realResult;
		//out.prediction = predict(false);
		out.errorValue = realResult - prediction;
		out.costValue = previousCost + Math.pow(out.errorValue, 2);
//		out.lastBias = this.bias;
//		out.lastWeights = this.weights;
		switch(this.type) {
			case perceptron:
				defaultRule(out.errorValue);
				break;
			case adalaine:
				defaultRule(out.errorValue);
//				if(this.step) {
//					out.prediction = stepFunction(out.prediction);
//				}
				break;
			default:
				System.out.println("Regra de aprendizagem não implementada");
		}
//		out.newWeights = this.weights;
//		out.newBias = this.bias;
		return out;
	}
	
	private void defaultRule(double errorValue) {
		for (int i=0; i< this.entries.size(); i++) {
			this.weights.set(i, this.weights.get(i)+ (this.learningRateW * errorValue * this.entries.get(i)));
			this.bias = this.bias + learningRateB * errorValue;
		}
	}
	
	
	
	//Função Aditiva
	public double aditive() {
		double out = 0;
		if(this.weights.size() != this.entries.size()) {
			generateWeightsAndBias();
		}
		for(int i=0; i<this.entries.size();i++) {
			out += this.entries.get(i) * this.weights.get(i);
		}
		return out + this.bias;
	}
	
	public double predict(boolean derivative) {
		double out = aditive();
		switch (this.activation) {
			case Linear:
				out = linear(out, derivative);
				break;
			case Sigmoid:
				out = sigmoid(out, derivative);
				break;
			case Tanh:
				out = tanH(out, derivative);
				break;
			case ReLU:
				out = reLU(out, derivative);
				break;
			case LeakyReLU:
				out = leakReLU(out, derivative);
				break;
			case eLU:
				out = eLU(out, derivative);
				break;
		}
		//Aplicação step;
		switch (this.type) {
			case perceptron:
				return  this.step?stepFunction(out):out;
			case adalaine:
				return out;
			case sigmoid:
				System.out.println("predição sigmoid não implementada.");
				return Double.MIN_VALUE;
			default:
				System.out.println("predição de tipo desconhecido não implementada.");
				return Double.MIN_VALUE;
		}
		
	}
	
	//Função Step
	public double stepFunction(double y) {
		return y>0?1:0;
	}
	
	// Funções de Ativiação
	private double linear(double x, boolean derivative) {
		return derivative?1:x;
	}
	
	private double sigmoid(double x, boolean derivative) {
		double out = 1/(1 + Math.exp(-x));
		return derivative?out*(1-out):out;
	}
	
	private double tanH(double x, boolean derivative) {
		double out = (Math.exp(x)-Math.exp(-x))/(Math.exp(x) + Math.exp(-x));
		return derivative?(1-Math.pow(out,2)):out;
	}
	
	private double reLU(double x, boolean derivative) {
		double out = Math.max(0, x);
		if(x<=0) {
			return derivative?0:out;
		}
		return derivative?1:out;
	}
	
	private double leakReLU(double x, boolean derivative) {
		if(x <=0) {
			return derivative?this.alfaReLU:this.alfaReLU*x;
		}
		return derivative?1:x;
	}
	
	private double eLU(double x, boolean derivative) {
		double out = this.alfaReLU * (Math.exp(x) - 1);
		if(x<=0) {
			return derivative?out+this.alfaReLU:out;
		}
		return derivative?1:x;
	}
	
	
}
