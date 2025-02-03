package br.com.qualiteti.qualitetirna.rna;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.util.FastMath;

public class CostOffLine {
	
	
	public static List<Double> meanSquareError_derivative(List<Double> predicts, List<Double> realValues) {
		List<Double> out = new ArrayList<>();
		double f;
		if(predicts.size() == realValues.size() && predicts.size() > 0) {
			f = -2.0/predicts.size();
			for (int i=0; i< predicts.size(); i++) {
				out.add(f*(realValues.get(i)-predicts.get(i)));
			}
		}
		return out;
	}
	
	public static double meanSquareError(List<Double> predicts, List<Double> realValues) {
		double out = Double.MAX_VALUE;
		double sum = 0.0;
		if(predicts.size() == realValues.size()) {
			for (int i=0; i< predicts.size(); i++) {
				sum += Math.pow((realValues.get(i) - predicts.get(i)),2);
			}
			out = sum/predicts.size();
		}
		return out;
	}

	public static double sumSquareError(List<Double> predicts, List<Double> realValues) {
		double out = Double.MAX_VALUE;
		double sum = 0.0;
		if(predicts.size() == realValues.size()) {
			for (int i=0; i< predicts.size(); i++) {
				sum += Math.pow((realValues.get(i) - predicts.get(i)),2);
			}
			out = sum;
		}
		return out;
	}

	public static List<Double> sumSquareError_derivative(List<Double> predicts, List<Double> realValues) {
		List<Double> out = new ArrayList<>();
		double f;
		if(predicts.size() == realValues.size() && predicts.size() > 0) {
			f = -1.0;
			for (int i=0; i< predicts.size(); i++) {
				out.add(f*(realValues.get(i)-predicts.get(i)));
			}
		}
		return out;
	}
	
	public static double crossEntropy(List<Double> predicts, List<Double> realValues) {
		double out = Double.MAX_VALUE;
		double sum = 0.0;
		if(predicts.size() == realValues.size()) {
			for(int i=0; i < predicts.size(); i++) {
				sum += -realValues.get(i) * Math.log(predicts.get(i)) - (1.0-realValues.get(i)) * Math.log(1.0-predicts.get(i)) ;
			}
			out = sum/predicts.size();
		}
		if(Double.isNaN(out)) {
			sum = sum +1.0 -1.0;
		}
		return out;
	}
	
	public static List<Double> crossEntropy_derivative(List<Double> predicts, List<Double> realValues) {
		List<Double> out = new ArrayList<>();
		double f;
		if(predicts.size() == realValues.size() && predicts.size() > 0) {
			f = 1.0/predicts.size();
			for(int i=0; i < predicts.size(); i++) {
				out.add((-1.0*(realValues.get(i)- predicts.get(i)))/(predicts.get(i)*(1.0-predicts.get(i)))*f);
			}
		}
		
		return out;
	}
	
	public static double SigmoidCrossEntropy(List<Double> predicts, List<Double> realValues) {
		double out = Double.MAX_VALUE;
		double sum = 0.0;
		List<Double> sig = sigmoid(predicts, false);
		if(sig.size() == realValues.size()) {
			for(int i=0; i < sig.size(); i++) {
				sum += -realValues.get(i) * Math.log(sig.get(i)) - (1.0-realValues.get(i)) * Math.log(1.0-sig.get(i)) ;
			}
			out = sum/sig.size();
		}
		
		return out;
	}
	
	public static List<Double> SigmoidCrossEntropy_derivative(List<Double> predicts, List<Double> realValues) {
		List<Double> out = new ArrayList<>();
		List<Double> sig = sigmoid(predicts, false);
		double f;
		if(sig.size() == realValues.size() && sig.size() > 0) {
			f = 1.0/sig.size();
			for(int i=0; i < predicts.size(); i++) {
				out.add((-1.0 * (realValues.get(i) - sig.get(i)))*f);
			}
		}
		
		return out;
	}
	
	public static double meanAbsolutError(List<Double> predicts, List<Double> realValues) {
		double out = Double.MAX_VALUE;
		double sum =0.0;
		if(predicts.size() == realValues.size()) {
			for(int i=0; i< predicts.size();i++) {
				sum += FastMath.abs(realValues.get(i) - predicts.get(i));
			}
			out = sum/predicts.size();
		}
		return out;
	}
	
	public static List<Double> meanAbsolutError_derivative(List<Double> predicts, List<Double> realValues) {
		List<Double> out = new ArrayList<>();
		if(predicts.size() == realValues.size()) {
			for(int i=0; i< predicts.size();i++) {
				if(realValues.get(i) > predicts.get(i)) {out.add(-1.0/predicts.size());}
				else if(realValues.get(i) > predicts.get(i)) {out.add(1.0/predicts.size());}
				else {out.add(0.0);}
			}
		}
		return out;
	}
	
	public static double sumAbsolutError(List<Double> predicts, List<Double> realValues) {
		double out = 0.0;
		if(predicts.size() == realValues.size()) {
			for(int i=0; i< predicts.size();i++) {
				out += realValues.get(i) - predicts.get(i);
			}
		}
		return out;
	}
	
	public static List<Double> sumAbsolutError_derivative(List<Double> predicts, List<Double> realValues) {
		List<Double> out = new ArrayList<>();
		if(predicts.size() == realValues.size()) {
			for(int i=0; i< predicts.size();i++) {
				if(realValues.get(i) > predicts.get(i)) {out.add(-1.0);}
				else if(realValues.get(i) < predicts.get(i)) {out.add(1.0);}
				else {out.add(0.0);}
			}
		}
		return out;
	}
	
	
	
	
	public static double softmaxNegLogLikelihood(List<Double> predicts, List<Double> realValesOneHot) {
		double out = Double.MAX_VALUE;
		int id =-1;
		double v = Double.MIN_VALUE;
		realValesOneHot = Encodding.convertToOneHotEncoding(realValesOneHot);
		List<Double> softMaxResults = softMax(predicts);
		for(int i=0; i<softMaxResults.size(); i++) {
			if(realValesOneHot.get(i) > v) {
				id = i;
				v = realValesOneHot.get(i);
			}
		}
		out = (-Math.log(softMaxResults.get(id)));  // /predicts.size();
		return out;
	}
	
	private static List<Double> negLogLikelihood_derivative(List<Double> predicts, List<Double> realValuesOneHot){
		List<Double> out = predicts;
		double v = Double.MIN_VALUE;
		int k = -1;
		for(int i=0; i<realValuesOneHot.size(); i++) {
			if(realValuesOneHot.get(i)>v) {
				v = realValuesOneHot.get(i);
				k = i;
			}
		}
		out.set(k, -1/out.get(k));
		return out;
	}
	
	public static List<Double> softmaxNegLogLikelihood_derivative(List<Double> predicts, List<Double> realValuesOneHot) {
		List<Double> y_softmax = softMax(predicts);
		int k = -1;
		double v = Double.MIN_VALUE;
		for(int i=0; i<realValuesOneHot.size(); i++) {
			if(realValuesOneHot.get(i) > v) {
				v = realValuesOneHot.get(i);
				k = i;
			}
		}
		List<Double> dLogNeg = negLogLikelihood_derivative(y_softmax, realValuesOneHot);
		List<Double> dSoftmax = softMax_derivative(predicts, realValuesOneHot);
		y_softmax.set(k,dLogNeg.get(k) * dSoftmax.get(k));
		dSoftmax = null;
		return y_softmax;
	}
	
	
	//Softmax não é uma função de custo, mas é usada em uma
//	public static double softMax(List<Double> predicts, double predict, boolean derivative ) {
//		double out;
//		double sum = 0;
//		
//		for(double p:predicts) {
//			sum += Math.exp(p);
//		}
//		
//		out = Math.exp(predict)/sum;
//		
//		return derivative?(out*(1-out)):out;
//		
//	}
	
	public static List<Double> softMax(List<Double> predicts ) {
		List<Double> out = new ArrayList<>();
		double sum = 0.0;
		for(double p:predicts) {
			sum += Math.exp(p);
		}
		for(Double p:predicts) {
			out.add(Math.exp(p)/sum);
		}
		return out;
	}
	
	public static List<Double> softMax_derivative(List<Double> predicts, List<Double> realValue ) {
		List<Double> temp = new ArrayList<>();
		int maxId = -1;
		double v = Double.MIN_VALUE;
		temp = softMax(predicts);
		for(int i=0; i<realValue.size(); i++) {
			if(realValue.get(i)> v) {
				v = realValue.get(i);
				maxId = i;
			}
		}
		temp.set(maxId, temp.get(maxId) * (1.0-temp.get(maxId)));
		return temp;
	}
	
	
	//MÉTODOS PRIVADOS
	private static List<Double> sigmoid(List<Double> zList, boolean derivative) {
		List<Double> out = new ArrayList<>();
		double v;
		for(double z:zList) {
			v =1.0/(1.0 + Math.exp(-z));
			out.add(derivative?v*(1-v):v);
		}
		return out;
	}
}
