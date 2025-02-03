package br.com.qualiteti.qualitetirna.common;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

public class MatrixExtras {
	public static double sumAll(RealMatrix matrix) {
		double sum = 0.0;
		for (double[] row : matrix.getData()) {
			for (double value : row) {
					sum += value; 
			} 
		}
		return sum;
	}
	
	public static RealMatrix elementwiseMultiply(RealMatrix a, RealMatrix b) {
		//Condição de existência
		if(a.getRowDimension() != b.getRowDimension() || a.getColumnDimension() != b.getColumnDimension()) {
			return null;
		}
		//Operação
		RealMatrix out = a;
		for(int r=0; r<a.getRowDimension(); r++) {
			for(int c=0; c<a.getColumnDimension(); c++) {
				out.setEntry(r, c,  a.getEntry(r, c) * b.getEntry(r, c));
			}
		}
		return out;
	}
	
	public static List<Double> vectorHadamardMultiply(List<Double> a, List<Double>b){
		if(a.size() != b.size()) {
			throw new ArrayIndexOutOfBoundsException("Os vetores para multplicação Hadamard devem ter o mesmo tamanho.");
		}
		List<Double> out = new ArrayList<>();
		for(int i=0; i<a.size(); i++) {
			out.add(a.get(i) * b.get(i));
		}
		return out;
	}
	
	public static RealMatrix elementewisePower(RealMatrix a, double power) {
		RealMatrix out = a;
		for(int r=0; r<a.getRowDimension(); r++) {
			for(int c=0; c<a.getColumnDimension(); c++) {
				out.setEntry(r, c,  FastMath.pow( a.getEntry(r, c), power)    );
			}
		}
		return out;
	}
}
