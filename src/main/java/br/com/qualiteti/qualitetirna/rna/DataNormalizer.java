package br.com.qualiteti.qualitetirna.rna;

import java.util.ArrayList;
import java.util.List;

public class DataNormalizer {
	/**
	 * Normaliza a entrada de dados entre -1 e 1
	 * 
	 * @param values Entrada de dados em uma lista bidimensional
	 * @return Retorna uma lista bidimensional de dados normalizados
	 */
	public static List<List<Double>> normalizeX(List<List<Double>> values){
		List<List<Double>> out = new ArrayList<>();
		List<Double> line;
		List<Double> min = new ArrayList<>();
		List<Double> max = new ArrayList<>();
		int nCols = 0;
		
		if(values != null) {
			if(!values.isEmpty()) {
				if(values.size() > 0) {
					if(values.get(0) != null) {
						if(!values.get(0).isEmpty()) {
							nCols = values.get(0).size();
						}
					}
				}
			}
		}
		
		for(int c=0; c<nCols; c++) {
			//Resgatando mínimo e máximo da coluna
			min.add(Double.MAX_VALUE);
			max.add(Double.MIN_VALUE);
			for(List<Double> l:values) {
				if(min.get(c) > l.get(c)) {min.set(c, l.get(c));}
				if(max.get(c) < l.get(c)) {max.set(c, l.get(c));}
			}
			//Normalizando saída
			for(int i=0; i< values.size(); i++) {
				if(c==0) {	//Criando o List de cada linha
					//line = Arrays.asList(     (2*((values.get(i).get(c) - min.get(c))/(max.get(c) - min.get(c))))-1        );
					line = new ArrayList<>();
					line.add((2*((values.get(i).get(c) - min.get(c))/(max.get(c) - min.get(c))))-1 );
					out.add(line);
				}
				else {		//Incluindo valor no List de cada linha
					out.get(i).add(  (2*((values.get(i).get(c) - min.get(c))/(max.get(c) - min.get(c))))-1  );
				}
			}
		}
		
		return out;
	}
	
	
	/**
	 * Normaliza a entrada de dados entre -1 e 1
	 * 
	 * @param values Dados em uma lista bidimensional
	 * @param minValue Valor mínimo do range de normalização
	 * @param maxValue Valor máximo do range de normalização
	 * @return Retorna uma lista bidimensional de dados normalizados
	 */
	public static List<List<Double>> normalizeX(List<List<Double>> values, double minValue, double maxValue){
		List<List<Double>> out = new ArrayList<>();
		List<Double> line;
		List<Double> min = new ArrayList<>();
		List<Double> max = new ArrayList<>();
		int nCols = 0;
		double diff = maxValue - minValue;
		double value;
		
		if(values != null) {
			if(!values.isEmpty()) {
				if(values.size() > 0) {
					if(values.get(0) != null) {
						if(!values.get(0).isEmpty()) {
							nCols = values.get(0).size();
						}
					}
				}
			}
		}
		
		for(int c=0; c<nCols; c++) {
			//Resgatando mínimo e máximo da coluna
			min.add(Double.MAX_VALUE);
			max.add(Double.MIN_VALUE);
			for(List<Double> l:values) {
				if(min.get(c) > l.get(c)) {min.set(c, l.get(c));}
				if(max.get(c) < l.get(c)) {max.set(c, l.get(c));}
			}
			//Normalizando saída
			for(int i=0; i< values.size(); i++) {
				value =  (diff*((values.get(i).get(c) - min.get(c))/(max.get(c) - min.get(c))))+minValue;
				
				if(c==0) {	//Criando o List de cada linha
					//line = Arrays.asList(     (2*((values.get(i).get(c) - min.get(c))/(max.get(c) - min.get(c))))-1        );
					line = new ArrayList<>();
					line.add(value );
					out.add(line);
				}
				else {		//Incluindo valor no List de cada linha
					out.get(i).add(  value  );
				}
			}
		}
		
		return out;
	}
	
	
	/**
	 * Normaliza uma lista unidimensional de dados
	 * 
	 * @param values  Lista unidimensional de dados de entrada
	 * @return        Lista unidimensional de dados normalizados
	 */
	public static List<Double> normalizeY(List<Double> values){
		List<Double> out = new ArrayList<>();
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		
		
		if(values == null) {
			return null;
		}
		if(values.isEmpty()) {
			return null;	
		}
		
		for(double v:values) {
			if(min > v) {min = v;}
			if(max < v) {max = v;}
		}
		for(double v:values) {
			out.add(   (2*((v - min)/(max - min)))-1   );
		}
		
		return out;
	}
	
	/**
	 * Normaliza uma lista unidimensional de dados
	 * 
	 * @param values  Lista unidimensional de dados de entrada
	 * @param minValue Valor mínimo do range de normalização
	 * @param maxValue Valor máximo do range de normalização
	 * @return        Lista unidimensional de dados normalizados
	 */
	public static List<Double> normalizeY(List<Double> values, double minValue, double maxValue){
		List<Double> out = new ArrayList<>();
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double diff = maxValue - minValue;
		
		
		if(values == null) {
			return null;
		}
		if(values.isEmpty()) {
			return null;	
		}
		
		for(double v:values) {
			if(min > v) {min = v;}
			if(max < v) {max = v;}
		}
		for(double v:values) {
			out.add(   (diff*((v - min)/(max - min)))+minValue   );
		}
		
		return out;
	}
}
