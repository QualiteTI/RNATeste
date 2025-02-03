package br.com.qualiteti.qualitetirna.rna;

import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnLayerType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.rna.definition.LayerDefinition;
import br.com.qualiteti.qualitetirna.rna.definition.NeuronDefinition;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronAdaline;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronFlex;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronPerceptron;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSigmoid;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSuper;

public class Layer {
	public EnLayerType layerType;
	public EnNeuronType neuronsType;
	public int netOrder;
	public List<NeuronSuper> neurons = new ArrayList<>();
	public double layerLearningRate = 1e2;
	public List<Double> uniformLayer = new ArrayList<>();
	public List<Double> forward = new ArrayList<>();
	public List<Double> backPropagationGradient = new ArrayList<>();
	
	public Layer(EnLayerType layerType, EnNeuronType neuronsType, int netOrder, int neuronsQuantity
			     , EnActivationType activationFunctionForFlex, ENCostFunctions costFunctionsForFlex) {
		this.layerType = layerType;
		this.neuronsType = neuronsType;
		this.netOrder = netOrder;
		
		for(int i=0; i<neuronsQuantity; i++) {
			switch(neuronsType) {
				case perceptron:
					this.neurons.add(new NeuronPerceptron());
					break;
				case adalaine:
					this.neurons.add(new NeuronAdaline());
					break;
				case sigmoid:
					this.neurons.add(new NeuronSigmoid());
					break;
				case flex:
					this.neurons.add(new NeuronFlex(activationFunctionForFlex, costFunctionsForFlex));
					break;
			}
			this.neurons.get(i).setLearningRate(layerLearningRate);
		}
	}
	
	public Layer (LayerDefinition def) {
		this.load(def);
	}
	
	public void load(LayerDefinition def) {
		this.layerType = def.layerType;
		this.neuronsType = def.neuronsType;
		this.netOrder = def.netOrder;
		this.layerLearningRate = def.layerLearningRate;
		this.neurons = new ArrayList<>();
		for(NeuronDefinition n:def.neuronsDefinitions) {
			NeuronSuper neuron = null;
			switch(n.type) {
				case perceptron:
					neuron = new NeuronPerceptron();
					break;
				case adalaine:
					neuron = new NeuronAdaline();
					break;
				case sigmoid:
					neuron = new NeuronSigmoid();
					break;
				case flex:
					neuron = new NeuronFlex(n.activationFunction, n.costFunction);
					break;
			}
			neuron.load(n);
			this.neurons.add(neuron);
			neuron = null;
		}
	}
	
	public LayerDefinition export() {
		LayerDefinition out = new LayerDefinition();
		out.layerType = this.layerType;
		out.neuronsType = this.neuronsType;
		out.netOrder = this.netOrder;
		out.layerLearningRate = this.layerLearningRate;
		out.neuronsDefinitions = new ArrayList<>();
		for(NeuronSuper n:this.neurons) {
			out.neuronsDefinitions.add(n.export());
		}
		return out;
	}
	
	public void setValues(List<List<Double>> values) {
		if(values.size() != this.neurons.size()) {
			throw new ArrayIndexOutOfBoundsException("Número de valores diferente do número de neurônios - Camada:" + this.netOrder);
		}
		for(int i=0; i < values.size(); i++) {
			this.neurons.get(i).entries = values.get(i);
		}
	}
	
	public void setWeight(List<List<Double>> values) {
		if(values.size() != this.neurons.size()) {
			throw new ArrayIndexOutOfBoundsException("Número de pesos diferente do número de neurônios - Camada:" + this.netOrder);
		}
		else if(values.get(0).size() != this.neurons.get(0).entries.size() && this.layerType == EnLayerType.input) {
			throw new ArrayIndexOutOfBoundsException("Quantidade de pesos por neurônio diferente do número de entradas por neurônio - Camada:" + this.netOrder);
		}
		for(int i=0; i < values.size(); i++) {
			this.neurons.get(i).weights = values.get(i);
		}
	}
	
	public void setBias(List<Double> values) {
		if(values.size() != this.neurons.size()) {
			throw new ArrayIndexOutOfBoundsException("Número de valores diferente do número de neurônios - Camada:" + this.netOrder);
		}
		for(int i=0; i < values.size(); i++) {
			this.neurons.get(i).bias = values.get(i);
		}
	}
	
	public void setLearningRateForLayer(double learningRate) {
		for(NeuronSuper n:this.neurons) {
			n.setLearningRate(learningRate);
		}
	}
	
	public void setStepFunction(boolean stepOn) {
		for(NeuronSuper n:this.neurons) {
			n.step = stepOn;
		}
	}
	
	public void generateWeightsAndBias(int nEntries) {
		for(NeuronSuper n:this.neurons) {
			n.generateWeightsAndBias(nEntries);
		}
	}
	
	public List<Double> getPredicts(){
		List<Double> out = new ArrayList<>();
		for(NeuronSuper n:this.neurons) {
			out.add(n.predict());
		}
		return out;
	}
	
	public List<Double> doForward(){
		List<Double> out = new ArrayList<>();
		for(int i=0; i< this.neurons.size(); i++) {
			out.add(this.neurons.get(i).predict());
		}
		this.forward = out;
		return out;
	}
	
	
	/**
	 * Calcula o gradiente da presente camada para a camada anterior
	 * Também atualiza os pesos e bias de todos os neurônios da camada atual
	 * 
	 * @param gradient - List<Double> representando o gradiente da camada posterior. Se for a última camada recebe a derivada da função de custo.
	 * @return - List<Double> retorna o gradiente para a camada anterior.
	 */
	public List<Double> doBackPropagation(List<Double> gradient) {
		List<Double> out = new ArrayList<>();
		double sum;
		
		if(gradient.size() != this.neurons.size()) {
			throw new ArrayIndexOutOfBoundsException("Nº de elementos do gradiente diferente do nº de neurônios da camada");
		}
		//Gerando novo gradiente de saída:
		List<Double> dAtivGrad = new ArrayList<>();
		for(int n=0; n<this.neurons.size(); n++) {
			dAtivGrad.add(this.neurons.get(n).getActivation_derivative() * gradient.get(n));
		}
		for(int c=0; c < this.neurons.get(0).weights.size(); c++) {
			sum = 0;
			for(int n=0; n<this.neurons.size(); n++) {
				sum += dAtivGrad.get(n) * this.neurons.get(n).weights.get(c);
			}
			out.add(sum);
		}
		
		
		
		backPropagationGradient = out;
		return out;
	}
	
	public void updateLayerWeigthsAndBias(List<Double> gradient) {
		List<Double> dAtivGrad = new ArrayList<>();
		for(int n=0; n<this.neurons.size(); n++) {
			dAtivGrad.add(this.neurons.get(n).getActivation_derivative() * gradient.get(n));
		}
		//Gerando novos pesos e bias para cada neurônio da camada.
				for(int n=0; n<this.neurons.size(); n++) {
					for(int w=0; w<this.neurons.get(n).weights.size(); w++) {
						this.neurons.get(n).weights.set(w, this.neurons.get(n).weights.get(w) - ( dAtivGrad.get(n) * this.neurons.get(n).entries.get(w) * this.neurons.get(n).getActivation_derivative() * this.neurons.get(n).learningRateW));
					}
					
					this.neurons.get(n).bias = this.neurons.get(n).bias - (dAtivGrad.get(n) * this.neurons.get(n).learningRateB);
				}
	}
	
}
