package br.com.qualiteti.qualitetirna.rna.neurons;

import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronDerivativeType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;
import br.com.qualiteti.qualitetirna.rna.definition.NeuronDefinition;

public class NeuronSuper {
	public EnNeuronType type;
	public EnActivationType activationFunction;
	public ENCostFunctions costFunction;
	public double bias = 0;
	public double alfaReLU = 0;
	public double learningRateW = 0.1;
	public double learningRateB = 0.1;
	public boolean step = false;
	public List<Double> entries = new ArrayList<>();
	public List<Double> weights = new ArrayList<>();
	
	
	public NeuronSuper(EnNeuronType type) {
		this.type = type;
	}
	
	public void load(NeuronDefinition def) {
		this.type = def.type;
		this.activationFunction = def.activationFunction;
		this.costFunction = def.costFunction;
		this.bias = def.bias;
		this.alfaReLU = def.alfaReLU;
		this.learningRateW = def.learningRateW;
		this.learningRateB = def.learningRateB;
		this.step = def.step;
		this.weights = def.weights;
	}
	
	public NeuronDefinition export() {
		NeuronDefinition out = new NeuronDefinition();
		out.type = this.type;
		out.activationFunction = this.activationFunction;
		out.costFunction = this.costFunction;
		out.bias = this.bias;
		out.alfaReLU = this.alfaReLU;
		out.learningRateW = this.learningRateW;
		out.learningRateB = this.learningRateB;
		out.step = this.step;
		out.weights = this.weights;
		return out;
	}
	
	public void setLearningRate(double learningRate) {
		this.learningRateW = learningRate;
		this.learningRateB = learningRate;
	}
	
	public void generateWeightsAndBias() {
		weights = new ArrayList<>();
		for(int i=0; i<this.entries.size();i++) {
			this.weights.add(2.0*Math.random()-1.0);
		}
		this.bias = 2.0*Math.random()-1.0;
	}
	
	public void generateWeightsAndBias(int nEntries) {
		this.weights = new ArrayList<>();
		for(int i=0; i<nEntries;i++) {
			this.weights.add(2.0*Math.random()-1.0);
		}
		this.bias = 2.0*Math.random()-1.0;
	}
	
	//Funções Aprendizagem
		public LearnResult learnOnLine(double realResult, double previousCost) {
			System.out.println("A função de aprendizagem deve ser especializada");
			return null;
		}
		
		public LearnResult learnOffLine(List<Double> realResults, List<Double> predicts, List<List<Double>> data) {
			System.out.println("A função de aprendizagem deve ser especializada");
			return null;
		}
		
	//Funções de predição
		public double predict() {
			double out = aditive();
			double min = -1.0;
			double max = 1.0;
			switch(this.activationFunction) {
				case Linear:
					out = linear(out, false);
					min = 0;
					break;
				case Sigmoid:
					out = sigmoid(out, false);
					break;
				case Tanh:
					out = tanH(out, false);
					break;
				case ReLU:
					out = reLU(out, false);
					break;
				case LeakyReLU:
					out = leakReLU(out, false);
					break;
				case eLU:
					out = eLU(out, false);
					break;
			}
			
			return this.step?this.stepFunction(min, max, out):out;
		}
		
		public List<Double> predict_derivative(EnNeuronDerivativeType derivativeType){
			List<Double> out = new ArrayList<>();
			double vAditive = aditive(); 
			switch(derivativeType) {
				case bias:
					switch(this.activationFunction) {
						case Linear:
							out.add(this.linear(vAditive, true));
							break;
						case Sigmoid:
							out.add(this.sigmoid(vAditive, true));
							break;
						case Tanh:
							out.add(this.tanH(vAditive, true));
							break;
						case ReLU:
							out.add(this.reLU(vAditive, true));
							break;
						case LeakyReLU:
							out.add(this.leakReLU(vAditive, true));
							break;
						case eLU:
							out.add(this.eLU(vAditive, true));
							break;
					}
					break;
				case entries:
					for(Double w:this.weights) {
						switch(this.activationFunction) {
							case Linear:
								out.add(this.linear(vAditive, true) * w);
								break;
							case Sigmoid:
								out.add(this.sigmoid(vAditive, true) * w);
								break;
							case Tanh:
								out.add(this.tanH(vAditive, true) * w);
								break;
							case ReLU:
								out.add(this.reLU(vAditive, true) * w);
								break;
							case LeakyReLU:
								out.add(this.leakReLU(vAditive, true) * w);
								break;
							case eLU:
								out.add(this.eLU(vAditive, true) * w);
								break;
						}
					}
					break;
				case weights:
					for(Double e:this.entries) {
						switch(this.activationFunction) {
							case Linear:
								out.add(this.linear(vAditive, true) * e);
								break;
							case Sigmoid:
								out.add(this.sigmoid(vAditive, true) * e);
								break;
							case Tanh:
								out.add(this.tanH(vAditive, true) * e);
								break;
							case ReLU:
								out.add(this.reLU(vAditive, true) * e);
								break;
							case LeakyReLU:
								out.add(this.leakReLU(vAditive, true) * e);
								break;
							case eLU:
								out.add(this.eLU(vAditive, true) * e);
								break;
						}
					}
					break;
				case adictive:
					switch(this.activationFunction) {
						case Linear:
							out.add(this.linear(vAditive, true));
							break;
						case Sigmoid:
							out.add(this.sigmoid(vAditive, true));
							break;
						case Tanh:
							out.add(this.tanH(vAditive, true));
							break;
						case ReLU:
							out.add(this.reLU(vAditive, true));
							break;
						case LeakyReLU:
							out.add(this.leakReLU(vAditive, true));
							break;
						case eLU:
							out.add(this.eLU(vAditive, true));
							break;
					}
			}
			return out;
		}
		
		
		
		//Função Aditiva
		protected double aditive() {
			double out = 0.0;
			if(this.weights.size() != this.entries.size()) {
				generateWeightsAndBias();
			}
			for(int i=0; i<this.entries.size();i++) {
				out += this.entries.get(i) * this.weights.get(i);
			}
			return out + this.bias;
		}
		
		/**
		 * Retorna a derivada da função aditiva em realação a um índice de Entrada/Peso
		 * 
		 * @param indice - Integer que informa o índice a ser usado como referência. em conjunto como próximo parâmetro define a base da derivada.
		 * @param byWeight - Booleano que infomra se a derivada é sobre o peso. Caso negativo é sobre a entrada
		 * @return Double informando o valor da derivada calculado
		 * 
		 * OBS: A derivada do Bias é sempre 1.
		 */
		protected double aditive_derivative(int indice, boolean byWeight) {
			if(indice < this.entries.size()) {
				return byWeight?this.weights.get(indice):this.entries.get(indice);
			}
			throw new  ArrayIndexOutOfBoundsException(  "Índice inválido");
		}
		
		
		
		//Funções de Custo
		protected double meanSquareError(double errorValue, double previousCost) {
			return previousCost + Math.pow(errorValue, 2);
		}
		
		
		//Funções Step
		public double stepFunction(double y) {
			return y>0.0?1.0:0.0;
		}
		
		public double stepFunction(double min, double max, double y) {
			double mid = (min + max)/2.0;
			return y>mid?max:min;
		}
		
		
		//Funções Internas de Aprendizagem
		protected void defaultRule(double errorValue) {
			for (int i=0; i< this.entries.size(); i++) {
				this.weights.set(i, this.weights.get(i)+ (this.learningRateW * errorValue * this.entries.get(i)));
				this.bias = this.bias + learningRateB * errorValue;
			}
		}
		
		// Funções de Ativiação
		public double getActivation_derivative() {
			double out = 0.0;
			switch(this.activationFunction) {
				case Linear:
					out = linear(aditive(),true);
					break;
				case Sigmoid:
					out = sigmoid(aditive(), true);
					break;
				case Tanh:
					out = tanH(aditive(), true);
					break;
				case ReLU:
					out = reLU(aditive(), true);
					break;
				case LeakyReLU:
					out = leakReLU(aditive(), true);
					break;
				case eLU:
					out = eLU(aditive(), true);
					break;
			}
			return out;
		}
		
		
		protected double linear(double x, boolean derivative) {
			return derivative?1.0:x;
		}
		
		protected double sigmoid(double z, boolean derivative) {
			double out = 1.0/(1.0 + Math.exp(-z));
			return derivative?out*(1.0-out):out;
		}
		
		protected double tanH(double x, boolean derivative) {
			double out = (Math.exp(x)-Math.exp(-x))/(Math.exp(x) + Math.exp(-x));
			return derivative?(1.0-Math.pow(out,2)):out;
		}
		
		protected double reLU(double x, boolean derivative) {
			double out = Math.max(0.0, x);
			if(x<=0.0) {
				return derivative?0.0:out;
			}
			return derivative?1.0:out;
		}
		
		protected double leakReLU(double x, boolean derivative) {
			if(x <=0.0) {
				return derivative?this.alfaReLU:this.alfaReLU*x;
			}
			return derivative?1:x;
		}
		
		protected double eLU(double x, boolean derivative) {
			double out = this.alfaReLU * (Math.exp(x) - 1.0);
			if(x<=0.0) {
				return derivative?out+this.alfaReLU:out;
			}
			return derivative?1.0:x;
		}

}
