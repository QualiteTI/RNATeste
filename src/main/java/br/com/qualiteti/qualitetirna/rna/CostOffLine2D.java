package br.com.qualiteti.qualitetirna.rna;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import br.com.qualiteti.qualitetirna.common.MatrixExtras;

public class CostOffLine2D {

	// Mean Square Error 
	public static double meanSquareError(RealMatrix predicts, RealMatrix realValues) {
		RealMatrix squaredDiffMatrix =  MatrixExtras.elementewisePower( realValues.subtract(predicts),2);
		return MatrixExtras.sumAll(squaredDiffMatrix) / (squaredDiffMatrix.getRowDimension() * squaredDiffMatrix.getColumnDimension() * 2); 
	}

	// Mean Square Error Derivative 
	public static RealMatrix meanSquareError_derivative(RealMatrix predicts, RealMatrix realValues) {
		RealMatrix diffMatrix = realValues.subtract(predicts);
		double factor = -1.0 / (diffMatrix.getRowDimension() * diffMatrix.getColumnDimension());
		return diffMatrix.scalarMultiply(factor);
	}
	
	
    // Sum Square Error
    public static double sumSquareError(RealMatrix predicts, RealMatrix realValues) {
    	RealMatrix squaredDiffMatrix = MatrixExtras.elementewisePower(realValues.subtract(predicts),2); 
        return MatrixExtras.sumAll(squaredDiffMatrix);
    }

    public static RealMatrix sumSquareError_derivative(RealMatrix predicts, RealMatrix realValues) {
    	return realValues.subtract(predicts).scalarMultiply(-1.0);
    }



    // Mean Absolute Error
    public static double meanAbsoluteError(RealMatrix predicts, RealMatrix realValues) {
    	RealMatrix diffMatrix = realValues.subtract(predicts);
    	return MatrixExtras.sumAll(diffMatrix)/(diffMatrix.getColumnDimension() * diffMatrix.getRowDimension());
    }

    public static RealMatrix meanAbsoluteError_derivative(RealMatrix predicts, RealMatrix realValues) {
    	RealMatrix diffMatrix = predicts.subtract(realValues);
    	for(int r=0; r<diffMatrix.getRowDimension(); r++) {
    		for(int c=0; c<diffMatrix.getColumnDimension(); c++) {
    			diffMatrix.setEntry(r, c, (1.0/(diffMatrix.getRowDimension() * diffMatrix.getColumnDimension())) +  (diffMatrix.getEntry(r, c)>0.0?1.0:(diffMatrix.getEntry(r, c)<0.0?-1.0:0.0) ));
    		}
    	}
        return diffMatrix;
    }
    
    // Helper: Sum Absolute Error
    public static double sumAbsoluteError(RealMatrix predicts, RealMatrix realValues) {
    	return MatrixExtras.sumAll(realValues.subtract(predicts));
    }

    public static RealMatrix sumAbsoluteError_derivative(RealMatrix predicts, RealMatrix realValues) {
    	RealMatrix diffMatrix = predicts.subtract(realValues);
    	for(int r=0; r<diffMatrix.getRowDimension(); r++) {
    		for(int c=0; c<diffMatrix.getColumnDimension(); c++) {
    			diffMatrix.setEntry(r, c, (diffMatrix.getEntry(r, c)>0.0?1.0:(diffMatrix.getEntry(r, c)<0.0?-1.0:0.0) ));
    		}
    	}
        return diffMatrix;

    }
    
    // Cross Entropy
    public static double crossEntropy(RealMatrix predicts, RealMatrix realValues) {
    	//Condição de existência
    	if(predicts.getRowDimension() != realValues.getRowDimension() || predicts.getColumnDimension() != realValues.getColumnDimension()) {
    		throw new ArrayIndexOutOfBoundsException("A matriz de predição tem tamanho diferente da matriz de dados reais");
    	}
    	//Operação
    	RealMatrix outM = realValues;
    	double yr; //Valor Real
    	double yp; //Predição
    	for(int r=0; r<realValues.getRowDimension(); r++) {
    		for(int c=0; c<realValues.getColumnDimension(); c++) {
    			yr = realValues.getEntry(r, c);
    			yp = predicts.getEntry(r, c);
    			
    			outM.setEntry(r, c, ((yr* FastMath.log(yp) ) + ((1.0 - yr) * FastMath.log((1-yp)))));;
    		}
    	}
        return MatrixExtras.sumAll(outM) / (realValues.getRowDimension() * realValues.getColumnDimension());
    }

    public static RealMatrix crossEntropy_derivative(RealMatrix predicts, RealMatrix realValues) {
    	//Condição de existência
    	if(predicts.getRowDimension() != realValues.getRowDimension() || predicts.getColumnDimension() != realValues.getColumnDimension()) {
    		return null;
    	}
    	//Operação
    	RealMatrix outM = realValues;
    	double yr; //Valor Real
    	double yp; //Predição
    	double factor = 1.0/(realValues.getRowDimension() * realValues.getColumnDimension());
    	for(int r=0; r<realValues.getRowDimension(); r++) {
    		for(int c=0; c<realValues.getColumnDimension(); c++) {
    			yr = realValues.getEntry(r, c);
    			yp = predicts.getEntry(r, c);
    			
    			outM.setEntry(r, c, (     ((-1.0*(yr-yp))/(yp*(1-yp)))*factor      ));;
    		}
    	}
    	
    	return outM;
    }	

    // Softmax
    public static RealVector softMax(RealVector predicts, RealVector realValues) {
    	
    	double sum = 0.0;
    	double[] outD = new double[predicts.getDimension()]; 
    	for(double p:predicts.toArray()) {
    		sum += FastMath.exp(p);
    	}
    	for(int i=0; i<predicts.getDimension(); i++) {
    		outD[i] = FastMath.exp(predicts.getEntry(i))/sum ;
    	}
    	return MatrixUtils.createRealVector(outD);
    }
    public static double softMax_derivative(RealVector predicts, RealVector realValues) { 
    	int id = -1;
    	double v = Double.MIN_VALUE;
    	RealVector sm = softMax(predicts, realValues); 
    	for(int i=0; i<realValues.getDimension(); i++) {
    		if(realValues.getEntry(i) > v) {
    			v = realValues.getEntry(i);
    			id = i;
    		}
    	}
    	return sm.getEntry(id) * (1-sm.getEntry(id));
    }
    
    public static RealMatrix softMax(RealMatrix predicts, RealMatrix realValues) {
    	RealMatrix out = MatrixUtils.createRealMatrix(predicts.getRowDimension(),predicts.getColumnDimension() );
    	for(int i=0; i<predicts.getRowDimension();i++) {
    		out.setRowVector(i, softMax(predicts.getRowVector(i),realValues.getRowVector(i)));
    	}
    	return out;
    }
    
    public static RealVector softMax_derivative(RealMatrix predicts, RealMatrix realValues) {
    	double[] out = new double[predicts.getRowDimension()];
    	for(int r=0; r<predicts.getRowDimension(); r++) {
    		out[r] = softMax_derivative(predicts.getRowVector(r),realValues.getRowVector(r));
    	}
    	return MatrixUtils.createRealVector(out);
    }

 
}

