package br.com.qualiteti.qualitetirna.common;

import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.regex.Pattern;



import java.nio.file.Files;

public class ClasseUtils {
	public static String soNumero(String entrada) {
		String saida = null;
		if(!entrada.isEmpty()) {
			saida = "";
			for(char c:entrada.toCharArray()) {
				if((int) c >= 48 && (int)c <= 57) {
					saida += c;
				}
			}
		}
		return saida;
	}
	
	public static boolean notBlankEmpty(String entrada) {
		if(entrada != null) if(!entrada.isEmpty()) {return !entrada.isBlank();}
		return false;
	}
	
	public static boolean webRequestStringIsNull(String value) {
		if(value == null) return true;
		if(value.isEmpty()) return true;
		if(value.trim().toUpperCase().equals("NULL")) return true;
		
		return false;
	}
	
	public static boolean checkFolder(String stPath) {
		try {
			Files.createDirectories(Paths.get(stPath));
		}
		catch(Exception ex) {
			return false;
		}
		return true;
	}
	
	public static String getStNow() {
		SimpleDateFormat formatador = new SimpleDateFormat("yyyy-MM-DD HH:mm:ss");
		String saida = formatador.format(new Date()).replace(" ", "T");
		return saida;
	}

	public static boolean isNumeric(String strNum) {
		Pattern pattern = Pattern.compile("-?\\d+(\\.\\d+)?");
	    if (strNum == null) {
	        return false; 
	    }
	    return pattern.matcher(strNum).matches();
	}

	public static boolean isPalavra(String value) {
        if (value == null || value.isEmpty()) {
            return false;
        }
        for (char c : value.toCharArray()) {
            if (Character.isLetter(c)) {
                return true;
            }
        }
        return false;
    }



//    public static TimeRange parseTimeRange(String input) {
//		DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss,SSS");
//        // Regex para capturar o padrão das horas
//        String regex = "(\\d{2}:\\d{2}:\\d{2},\\d{3}) --> (\\d{2}:\\d{2}:\\d{2},\\d{3})";
//        Pattern pattern = Pattern.compile(regex);
//        Matcher matcher = pattern.matcher(input);
//
//        if (matcher.find()) {
//            // Extrai as duas partes de tempo da string
//            String startTimeString = matcher.group(1);
//            String endTimeString = matcher.group(2);
//
//            // Converte as strings para LocalTime
//            LocalTime startTime = LocalTime.parse(startTimeString, TIME_FORMATTER);
//            LocalTime endTime = LocalTime.parse(endTimeString, TIME_FORMATTER);
//
//            // Retorna uma nova instância de TimeRange
//            return new TimeRange(startTime, endTime);
//        } else {
//            throw new IllegalArgumentException("Formato de entrada inválido: " + input);
//        }
//    }
}
