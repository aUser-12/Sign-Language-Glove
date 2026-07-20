const int   FLEX_PINS[5]      = {A0, A1, A2, A3, A4};
const int   BAUD_RATE         = 115200;
const int   SAMPLE_DELAY_MS   = 20;       


const int   CALIB_SAMPLES     = 100;      
const float DRIFT_THRESHOLD   = 15.0;     
const float DRIFT_ALPHA       = 0.002;    
const float NOISE_ALPHA       = 0.1;      
const float SPIKE_MULTIPLIER  = 4.0;      
const int   RECALIB_INTERVAL  = 30000;    
const int   STILL_WINDOW      = 25;  
const float STILL_THRESHOLD   = 8.0; 

float baseline[5];        
float smoothed[5];        
float noise[5];           
float lastValid[5];       
bool  sensorOk[5];     


float recentReadings[5][STILL_WINDOW];
int   ringIdx = 0;

unsigned long lastRecalibCheck = 0;
bool calibrated = false;



float readRaw(int pin) {
 
  long sum = 0;
  for (int i = 0; i < 4; i++) sum += analogRead(pin);
  return sum / 4.0;
}

float variance(float* arr, int n) {
  float mean = 0;
  for (int i = 0; i < n; i++) mean += arr[i];
  mean /= n;
  float var = 0;
  for (int i = 0; i < n; i++) var += (arr[i] - mean) * (arr[i] - mean);
  return var / n;
}

bool sensorIsStill(int s) {

  float col[STILL_WINDOW];
  for (int i = 0; i < STILL_WINDOW; i++) col[i] = recentReadings[s][i];
  return variance(col, STILL_WINDOW) < (STILL_THRESHOLD * STILL_THRESHOLD);
}



void calibrate() {
  Serial.println("STATUS:CALIBRATING");

  for (int s = 0; s < 5; s++) {
    float sum = 0;
    float sumSq = 0;
    for (int i = 0; i < CALIB_SAMPLES; i++) {
      float r = readRaw(FLEX_PINS[s]);
      sum   += r;
      sumSq += r * r;
      delay(5);
    }
    float mean = sum / CALIB_SAMPLES;
    float var  = (sumSq / CALIB_SAMPLES) - (mean * mean);

    baseline[s]   = mean;
    smoothed[s]   = mean;
    lastValid[s]  = mean;
    noise[s]      = max(sqrt(var), 1.0); 
    sensorOk[s]   = true;

  
    for (int i = 0; i < STILL_WINDOW; i++) recentReadings[s][i] = mean;
  }

  calibrated = true;
  Serial.println("STATUS:CALIBRATED");
}



float processSensor(int s, float raw) {

  
  float deviation = abs(raw - smoothed[s]);
  float spikeLimit = SPIKE_MULTIPLIER * max(noise[s], 3.0);

  if (deviation > spikeLimit) {
    noise[s] = noise[s] * (1 - NOISE_ALPHA) + deviation * NOISE_ALPHA;
    Serial.print("STATUS:SPIKE:");
    Serial.println(s);
    return lastValid[s];
  }


  noise[s] = noise[s] * (1 - NOISE_ALPHA) + deviation * NOISE_ALPHA;

 
  float smoothAlpha = 0.3;
  smoothed[s] = smoothed[s] * (1 - smoothAlpha) + raw * smoothAlpha;
  lastValid[s] = smoothed[s];


  recentReadings[s][ringIdx] = smoothed[s];

 
  if (sensorIsStill(s)) {
    float drift = smoothed[s] - baseline[s];
    if (abs(drift) > DRIFT_THRESHOLD) {
      Serial.print("STATUS:DRIFT:");
      Serial.print(s);
      Serial.print(":");
      Serial.println(drift, 1);
    }

 
    baseline[s] = baseline[s] * (1 - DRIFT_ALPHA) + smoothed[s] * DRIFT_ALPHA;
  }


  sensorOk[s] = (raw > 5 && raw < 1018);
  if (!sensorOk[s]) {
    Serial.print("STATUS:FAULT:");
    Serial.println(s);
  }

  return smoothed[s];
}



void printReadings(float* readings) {
  Serial.print("DATA:");
  for (int s = 0; s < 5; s++) {
    float corrected = readings[s] - baseline[s];
    Serial.print(corrected, 1);
    if (s < 4) Serial.print(",");
  }
  Serial.println();
}



void checkAutoRecalib() {  
  bool allStill = true;
  for (int s = 0; s < 5; s++) {
    if (!sensorIsStill(s)) { allStill = false; break; }
  }

  if (allStill) {
    for (int s = 0; s < 5; s++) {
      float drift = smoothed[s] - baseline[s];
      if (abs(drift) > DRIFT_THRESHOLD) { 
        baseline[s] += drift * 0.1;
      }
    }
    Serial.println("STATUS:RECALIB_APPLIED");
  }
}



void setup() {
  Serial.begin(BAUD_RATE);
  for (int s = 0; s < 5; s++) pinMode(FLEX_PINS[s], INPUT);
  Serial.println("STATUS:BOOT");
  Serial.println("STATUS:HOLD_STILL_FOR_CALIBRATION");
  delay(1000);
  calibrate();
}

void loop() {
  float readings[5];

  for (int s = 0; s < 5; s++) {
    float raw = readRaw(FLEX_PINS[s]);
    readings[s] = processSensor(s, raw);
  }

  ringIdx = (ringIdx + 1) % STILL_WINDOW;
  printReadings(readings);

  
  if (millis() - lastRecalibCheck > RECALIB_INTERVAL) {
    checkAutoRecalib();
    lastRecalibCheck = millis();
  }

  delay(SAMPLE_DELAY_MS);
}
