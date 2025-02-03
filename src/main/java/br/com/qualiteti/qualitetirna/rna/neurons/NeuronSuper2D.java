package br.com.qualiteti.qualitetirna.rna.neurons;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class NeuronSuper2D {
    public EnNeuronType type;
    public EnActivationType activationFunction;
    public ENCostFunctions costFunction;
    public double bias = 0;
    public double alfaReLU = 0;
    public double learningRateW = 0.1;
    public double learningRateB = 0.1;
    public boolean step = false;
    public RealMatrix entries; // Vetor bidimensional para entradas
    public RealMatrix weights; // Vetor bidimensional para pesos

    public NeuronSuper2D(EnNeuronType type) {
        this.type = type;
    }

    public void setLearningRate(double learningRate) {
        this.learningRateW = learningRate;
        this.learningRateB = learningRate;
    }
    
    /**
     * Define a entrada do neurônio. Se a entrada não for bidimensional, entrar os dados como linhas (nX1) em um vetor de 1 coluna.
     * @param entries
     */
    public void setEntries(RealMatrix entries) {
    	this.entries = entries;
    }
    
    public void setEntries(RealVector entriesVector) {
    	this.entries = MatrixUtils.createColumnRealMatrix(entriesVector.toArray());
    }

    public void generateWeightsAndBias(boolean bidimensionalWeights ) {
    	
    	if(bidimensionalWeights) {
    		this.weights = MatrixUtils.createRealMatrix(this.entries.getRowDimension(), this.entries.getColumnDimension());
    	}
    	else {
    		this.weights = MatrixUtils.createRealMatrix(this.entries.getRowDimension(), 1);
    	}
    	
    	for(int r=0; r<this.weights.getRowDimension(); r++) {
    		for(int c=0; r<this.weights.getColumnDimension(); c++) {
    			this.weights.setEntry(r, c, (2* Math.random() -1));
    		}
    	}
        this.bias = 2 * Math.random() - 1;
    }
    
    // Função Aditiva
    protected double[] aditive() {
    	
    	
    	
    	return null;
//        double[] out = new double[entries.getRowDimension()];
//        if (weights == null || weights.length != entries.length || weights[0].length != entries[0].length) {
//            generateWeightsAndBias();
//        }
//
//        for (int i = 0; i < entries.length; i++) {
//            double sum = 0;
//            for (int j = 0; j < entries[i].length; j++) {
//                sum += entries[i][j] * weights[i][j];
//            }
//            out[i] = sum + bias;
//        }
//        return out;
    }
    

    // Funções de predição
    public double[] predict() {
        double[] aditiveResult = aditive();
        double[] out = new double[aditiveResult.length];
        double min = -1;
        double max = 1;

        for (int i = 0; i < aditiveResult.length; i++) {
            double value = aditiveResult[i];
            switch (this.activationFunction) {
                case Linear -> value = linear(value, false);
                case Sigmoid -> value = sigmoid(value, false);
                case Tanh -> value = tanH(value, false);
                case ReLU -> value = reLU(value, false);
                case LeakyReLU -> value = leakReLU(value, false);
                case eLU -> value = eLU(value, false);
            }
            out[i] = this.step ? stepFunction(min, max, value) : value;
        }
        return out;
    }

//    public double[][] predict_derivative(EnNeuronDerivativeType derivativeType) {
//        double[][] out = new double[entries.length][entries[0].length];
//        double[] aditiveResult = aditive();
//
//        for (int i = 0; i < entries.length; i++) {
//            for (int j = 0; j < entries[i].length; j++) {
//                double vAditive = aditiveResult[i];
//                switch (derivativeType) {
//                    case bias -> out[i][j] = getActivationDerivative(vAditive);
//                    case entries -> out[i][j] = getActivationDerivative(vAditive) * weights[i][j];
//                    case weights -> out[i][j] = getActivationDerivative(vAditive) * entries[i][j];
//                }
//            }
//        }
//        return out;
//    }



    // Funções de ativação
    protected double getActivationDerivative(double x) {
        return switch (this.activationFunction) {
            case Linear -> linear(x, true);
            case Sigmoid -> sigmoid(x, true);
            case Tanh -> tanH(x, true);
            case ReLU -> reLU(x, true);
            case LeakyReLU -> leakReLU(x, true);
            case eLU -> eLU(x, true);
        };
    }

    protected double linear(double x, boolean derivative) {
        return derivative ? 1 : x;
    }

    protected double sigmoid(double z, boolean derivative) {
        double out = 1 / (1 + Math.exp(-z));
        return derivative ? out * (1 - out) : out;
    }

    protected double tanH(double x, boolean derivative) {
        double out = (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
        return derivative ? (1 - Math.pow(out, 2)) : out;
    }

    protected double reLU(double x, boolean derivative) {
        return derivative ? (x > 0 ? 1 : 0) : Math.max(0, x);
    }

    protected double leakReLU(double x, boolean derivative) {
        return derivative ? (x > 0 ? 1 : alfaReLU) : (x > 0 ? x : alfaReLU * x);
    }

    protected double eLU(double x, boolean derivative) {
        if (x > 0) return derivative ? 1 : x;
        double elu = alfaReLU * (Math.exp(x) - 1);
        return derivative ? elu + alfaReLU : elu;
    }

    // Funções auxiliares
    public double stepFunction(double y) {
        return y > 0 ? 1 : 0;
    }

    public double stepFunction(double min, double max, double y) {
        double mid = (min + max) / 2.0;
        return y > mid ? max : min;
    }
}

