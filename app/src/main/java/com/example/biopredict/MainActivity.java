package com.example.biopredict;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    EditText inputFieldPregnancies;
    EditText inputFieldGlucose;
    EditText inputFieldBloodPressure;
    EditText inputFieldSkinThickness;

    Button predictBtn;
    TextView resultTV;

    Interpreter interpreter;

    // Valores de normalización (deberías actualizar con los valores usados en Python)
    private final float[] minValues = {0, 78, 0, 0};
    private final float[] maxValues = {10, 197, 96, 47};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // Cargar el modelo
        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Inicialización de los campos de entrada
        inputFieldPregnancies = findViewById(R.id.editTextPregnancies);
        inputFieldGlucose = findViewById(R.id.editTextGlucose);
        inputFieldBloodPressure = findViewById(R.id.editTextBloodPressure);
        inputFieldSkinThickness = findViewById(R.id.editTextSkinThickness);

        predictBtn = findViewById(R.id.button);
        resultTV = findViewById(R.id.textView);

        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Verificar que los campos no estén vacíos
                if (areInputsValid()) {
                    try {
                        // Obtener los valores de entrada y normalizarlos
                        float pregnanciesValue = normalize(Float.parseFloat(inputFieldPregnancies.getText().toString()), minValues[0], maxValues[0]);
                        float glucoseValue = normalize(Float.parseFloat(inputFieldGlucose.getText().toString()), minValues[1], maxValues[1]);
                        float bloodPressureValue = normalize(Float.parseFloat(inputFieldBloodPressure.getText().toString()), minValues[2], maxValues[2]);
                        float skinThicknessValue = normalize(Float.parseFloat(inputFieldSkinThickness.getText().toString()), minValues[3], maxValues[3]);

                        // Crear la matriz de entrada
                        float[][] inputs = new float[1][4];
                        inputs[0][0] = pregnanciesValue;
                        inputs[0][1] = glucoseValue;
                        inputs[0][2] = bloodPressureValue;
                        inputs[0][3] = skinThicknessValue;

                        // Realizar la inferencia
                        float result = doInference(inputs);

                        // Desnormalizar el resultado
                        float desnormalizedResult = desnormalize(result, 0.134f, 2.288f); // Cambia estos valores a los min y max de y

                        // Mostrar resultado
                        resultTV.setText("Result: " + desnormalizedResult);

                    } catch (NumberFormatException e) {
                        resultTV.setText("Invalid input. Please enter numeric values.");
                    } catch (Exception e) {
                        resultTV.setText("An error occurred.");
                    }
                } else {
                    resultTV.setText("Please enter valid inputs.");
                }
            }
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private boolean areInputsValid() {
        // Verifica que los campos no estén vacíos y que sean números válidos
        return !inputFieldPregnancies.getText().toString().isEmpty() &&
                !inputFieldGlucose.getText().toString().isEmpty() &&
                !inputFieldBloodPressure.getText().toString().isEmpty() &&
                !inputFieldSkinThickness.getText().toString().isEmpty();
    }

    private float normalize(float value, float min, float max) {
        return (value - min) / (max - min);
    }

    private float desnormalize(float value, float min, float max) {
        return value * (max - min) + min;
    }

    public float doInference(float[][] input) {
        float[][] output = new float[1][1];
        interpreter.run(input, output);
        return output[0][0];
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("MIDTERM_linear.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long length = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }
}

