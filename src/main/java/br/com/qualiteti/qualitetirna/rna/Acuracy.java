package br.com.qualiteti.qualitetirna.rna;

import java.util.List;

public class Acuracy {
	public static double acuracy(List<Double> realValues, List<Double> predictions, boolean applyStep) {
		if(realValues.size() != predictions.size()) {
			return -1;
		}
		double c = 0; //Corrects
		double i = 0;  //Incorrects
		for(int k=0; k<realValues.size(); k++) {
			if((applyStep? stepFunction(predictions.get(k)):predictions.get(k)) == realValues.get(k)) {
				c++;
			}
			else {
				i++;
			}
		}
		return c/(c+i);	
	}
	
	//Função Step
	private static double stepFunction(double y) {
		return y>=0.5?1:0;
	}
}
