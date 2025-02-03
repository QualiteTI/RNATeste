package br.com.qualiteti.qualitetirna.rna;

import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnLayerType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.OperationalizableLayer;
import br.com.qualiteti.qualitetirna.common.types.OperationalizableNet;
import br.com.qualiteti.qualitetirna.common.types.OperationalizableNeuron;
import br.com.qualiteti.qualitetirna.rna.definition.LayerDefinition;
import br.com.qualiteti.qualitetirna.rna.definition.NetDefinition;

public class NeuralNet {
	public int totalLayers = 0;
	public List<Layer> layers = new ArrayList<>();
	public ENCostFunctions costFunction;
	public List<Double> realValues;
	private boolean entryLayerSeted = false;
	private boolean outLayerSeted = false;
	
	//CONFIGURAÇÃO
	public NeuralNet(ENCostFunctions costFunction) {
		this.costFunction = costFunction;
	}
	
	public NeuralNet(NetDefinition def) {
		this.load(def);
	}
	
	public NeuralNet(List<Double> realValues, ENCostFunctions costFunction) {
		this.realValues = realValues;
		this.costFunction = costFunction;
	}
	
	public NeuralNet(NetDefinition def, List<Double> realValues) {
		this.load(def);
		this.realValues = realValues;
		this.layers.get(0).setValues(replicateValues(realValues, this.layers.get(0).neurons.size()));
	}
	
	public void load(NetDefinition def) {
		this.costFunction = def.costFunction;
		this.layers = new ArrayList<>();
		for (LayerDefinition d:def.layersDefinitions) {
			this.layers.add(new Layer(d));
		}
	}
	
	public NetDefinition export() {
		NetDefinition out = new NetDefinition();
		out.costFunction = this.costFunction;
		for(Layer l:this.layers) {
			out.layersDefinitions.add(l.export());
		}
		return out;
	}
	
  	public void setEntryLayer(EnNeuronType neuronsType,  int neuronsQuantity
			     , EnActivationType activationFunctionForFlex, ENCostFunctions costFunctionsForFlex) {
		Layer entryLayer = new Layer(EnLayerType.input, neuronsType, 0, neuronsQuantity, activationFunctionForFlex, costFunctionsForFlex);
		if(this.layers.isEmpty()) {
			layers.add(entryLayer);
		}
		else {
			if(this.layers.get(0).layerType == EnLayerType.input) {
				this.layers.set(0, entryLayer);
			}
			else {
				this.layers.add(0, entryLayer);
			}
		}
		verifyLayersOrder();
		entryLayerSeted = true;
	}
	
	public void setOutLayer(EnNeuronType neuronsType,  int neuronsQuantity
		     , EnActivationType activationFunctionForFlex, ENCostFunctions costFunctionsForFlex) {
		int order = 1;
		Layer outLayer = new Layer(EnLayerType.output, neuronsType, order, neuronsQuantity, activationFunctionForFlex, costFunctionsForFlex);
		
		if(this.layers.isEmpty()) {
			this.layers.add(outLayer);
		}
		else {
			if(this.layers.get(this.layers.size()-1).layerType == EnLayerType.output) {
				this.layers.set(this.layers.size()-1, outLayer);
			}
			else {
				this.layers.add(outLayer);
			}
		}
		verifyLayersOrder();
		outLayerSeted = true;
	}
	
	/**
	 * 
	 * @param neuronsType - EnNeuronType - Define o tipo de neurônio da camada
	 * @param neuronsQuantity - int - Define o número de neurônios da camada
	 * @param activationFunctionForFlex - EnActivationType - No caso de uso de neurônio flex, define a função de ativação. Se o neurônio da camada não for flex, utilize NULL
	 * @param costFunctionsForFlex - EnCostFunction - No caso de uso de neurônio flex, define a função de custo. Se o neurônio da camada não for flex, utilize NULL
	 * @return - int - Retorna o número de ordem da camada dentro da Rede Neural.
	 */
	public int addOcultLayer(EnNeuronType neuronsType, int neuronsQuantity
		     , EnActivationType activationFunctionForFlex, ENCostFunctions costFunctionsForFlex) {
		int order =1;
		
		Layer layer = new Layer(EnLayerType.ocult, neuronsType, order, neuronsQuantity, activationFunctionForFlex, costFunctionsForFlex);
		
		if(this.layers.isEmpty()) {
			order = 0;
			this.layers.add(layer);
		}
		else {
			if(this.layers.get(this.layers.size()-1).layerType == EnLayerType.output) {
				order = this.layers.size() -1;
				this.layers.add(order, layer);
			}
			else {
				order = this.layers.size();
				this.layers.add(layer);
			}
		}
		verifyLayersOrder();
		
		return order;
	}
	
	public void setLearningRateForEntireNet(double learningRate) {
		for(Layer l:this.layers) {
			l.setLearningRateForLayer(learningRate);
		}
	}
	
	public void generateWeightsAndBias(int nEntries) {
		this.layers.get(0).generateWeightsAndBias(nEntries);
		for(int i=1;i<this.layers.size();i++) {
			this.layers.get(i).generateWeightsAndBias(this.layers.get(i-1).neurons.size());
		}
	}
	
	
	//VERIFICAÇÕES
	/**
	 * Este método verifica se a rede neural está operacional. Isso não significa que esteja de acordo com o projetado.
	 * 
	 * @return OperationalizableNet - A função isOk retorna o status geral da rede neural
	 * Se a lista de camadas problemáticas (problematicLayers) estiver vazia, todas as camadas estão ok. Se possuir problemas, estes são listados.
	 */
	public OperationalizableNet verifyNet() {
		OperationalizableNet out = new OperationalizableNet();
		out.hasInput = entryLayerSeted;
		out.hasOutPut = outLayerSeted;
		out.problematicLayers = verifyLayers();
		
		return out;
	}

	//Funções de custo da rede
	public double getTotalCost(List<Double> predicts) {
		double totalCost = Double.MAX_VALUE;
		switch(this.costFunction) {
			case crossEntropy:
				totalCost = CostOffLine.crossEntropy(predicts, this.realValues);
				break;
			case meanAbsoluteError:
				totalCost = CostOffLine.meanAbsolutError(predicts, this.realValues);
				break;
			case meanSquareError:
				totalCost = CostOffLine.meanSquareError(predicts, this.realValues);
				break;
			case sumAbsolutErro:
				totalCost = CostOffLine.sumAbsolutError(predicts, this.realValues);
				break;
			case sumSquareError:
				totalCost = CostOffLine.sumSquareError(predicts, this.realValues);
				break;
			case negLogLikelihood:
				totalCost = CostOffLine.softmaxNegLogLikelihood(predicts, this.realValues);
				break;
			case sigmoidCrossEntropy:
				totalCost = CostOffLine.SigmoidCrossEntropy(predicts, this.realValues);
				break;
		}
		return totalCost;
	}
	

	public List<Double> getTotalCost_derivative(List<Double> predicts) {
		List<Double> totalCostDerivative = new ArrayList<>();
		switch(this.costFunction) {
			case crossEntropy:
				totalCostDerivative = CostOffLine.crossEntropy_derivative(predicts, this.realValues);
				break;
			case meanAbsoluteError:
				totalCostDerivative = CostOffLine.meanAbsolutError_derivative(predicts, this.realValues);
				break;
			case meanSquareError:
				totalCostDerivative = CostOffLine.meanSquareError_derivative(predicts, this.realValues);
				break;
			case sumAbsolutErro:
				totalCostDerivative = CostOffLine.sumAbsolutError_derivative(predicts, this.realValues);
				break;
			case sumSquareError:
				totalCostDerivative = CostOffLine.sumSquareError_derivative(predicts, this.realValues);
				break;
			case negLogLikelihood:
				totalCostDerivative = CostOffLine.softmaxNegLogLikelihood_derivative(predicts, this.realValues);
				break;
			case sigmoidCrossEntropy:
				totalCostDerivative = CostOffLine.SigmoidCrossEntropy_derivative(predicts, this.realValues);
				break;
		}
		return totalCostDerivative;
	}
	
	
	//OPERAÇÕES
	
	
	public void setEntryValues(List<List<Double>> values) {
		this.layers.get(0).setValues(values);
	}
	
	
	
	/**
	 * Faz a operação de Forward da rede inteira
	 * 
	 * @return - double - Retorna o custo da rede inteira, baseada na função de custo derinida para a rede
	 */
	public double doForward() {
		double totalCost = 0;
		
		
		for (int i = 1; i < this.layers.size(); i++) {
			this.layers.get(i).uniformLayer = this.layers.get(i-1).doForward();
			this.layers.get(i).setValues(replicateValues(this.layers.get(i).uniformLayer, this.layers.get(i).neurons.size()));
		}
		//Executando a camada de saída:
		totalCost = getTotalCost(this.layers.get(this.layers.size()-1).doForward());
		
		return totalCost;
	}
	public double doForward(List<List<Double>> entryValues) {
		double totalCost = 0;
		
		this.setEntryValues(entryValues);
		
		for (int i = 1; i < this.layers.size(); i++) {
			this.layers.get(i).uniformLayer = this.layers.get(i-1).doForward();
			this.layers.get(i).setValues(replicateValues(this.layers.get(i).uniformLayer, this.layers.get(i).neurons.size()));
		}
		//Executando a camada de saída:
		totalCost = getTotalCost(this.layers.get(this.layers.size()-1).doForward());
		
		return totalCost;
	}
	
	/**
	 * Faz a operação de Backpropagation para todas as redes da última camada para a priméira (lógico).
	 * Neste processo, os pesos e bias de todos os neurônios e todas as camada são atualizadas
	 * Atenção para o LearningRate configurada 
	 * 
	 * @return  List<Double>  Retorna o gradiente da primeira camada.
 	 */
	public List<Double> doBackPropagationEntireNet () {
		List<Double> gradient;
		List<Double> newGradient;
		
		//Obtendo o gradiente da função de custo
		gradient = getTotalCost_derivative(this.layers.get(this.layers.size()-1).getPredicts());
//		print("predição da Net:" + this.layers.get(this.layers.size()-1).getPredicts());
//		print("Softmax da predição da Net:" + CostOffLine.softMax(this.layers.get(this.layers.size()-1).getPredicts()));
//		print("gradienteCusto:" + gradient);
		
		for(int l = this.layers.size()-1; l>=0; l--) {
			newGradient = this.layers.get(l).doBackPropagation(gradient);
			this.layers.get(l).updateLayerWeigthsAndBias(gradient);
			gradient = newGradient;
//			print("gradiente" + l + ":" + gradient);
		}
		return gradient;
	}

	public List<Double> getNetPredict(List<Double> entry, boolean bSoftmax){
		List<Double> out;
		
		this.doForward();
		out = this.layers.get(this.layers.size()-1).getPredicts();
		return bSoftmax?CostOffLine.softMax(out):out;
	}

	public List<Double> getNetPredict(boolean bSoftmax){
		List<Double> out;
		this.doForward();
		out = this.layers.get(this.layers.size()-1).getPredicts();
		return bSoftmax?CostOffLine.softMax(out):out;
	}
	
	//MÉTODOS PRIVADOS
//	private void print(String text) {
//		System.out.println(text);
//	}
	
 	public List<OperationalizableLayer> verifyLayers(){
		List<OperationalizableLayer> out = new ArrayList<>();
		OperationalizableLayer checkingLayer;
		OperationalizableNeuron checkingNeuron;
		
		for(Layer l:this.layers) {
			checkingLayer = new OperationalizableLayer();
			checkingLayer.netOrder = l.netOrder;
			checkingLayer.type = l.layerType;
			checkingLayer.hasNeurons = !l.neurons.isEmpty();
			for(int i=0; i<l.neurons.size(); i++) {
				checkingNeuron = new OperationalizableNeuron(i, l.neurons.get(i));
				if(!checkingNeuron.isOk(l.layerType == EnLayerType.input).value) {
					checkingLayer.prolematicsNeurons.add(checkingNeuron);
				}
				
			}
			if(!checkingLayer.isOk()) {
				out.add(checkingLayer);
			}
		}
		
		
		return out;
	}
	
	private void verifyLayersOrder() {
		if(!this.layers.isEmpty()) {
			for(int i=0; i<this.layers.size(); i++) {
				this.layers.get(i).netOrder = i;
			}
		}
		this.totalLayers = this.layers.size();
	}
	
	private List<List<Double>> replicateValues(List<Double> previousLayerOutput, int nextLayerNeurons){
		List<List<Double>> out = new ArrayList<>();
		for (int i=0; i< nextLayerNeurons; i++) {
			out.add(previousLayerOutput);
		}
		return out;
	}

	
}
