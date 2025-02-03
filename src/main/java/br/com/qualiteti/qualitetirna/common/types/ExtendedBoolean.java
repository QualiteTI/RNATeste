package br.com.qualiteti.qualitetirna.common.types;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
public class ExtendedBoolean {
	public boolean value;
	public String message;
	
	public void setValues(boolean value, String message) {
		this.value = value;
		this.message = message;
	}
}
