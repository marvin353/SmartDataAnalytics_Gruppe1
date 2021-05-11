* Encoding: UTF-8.

NPAR TESTS
  /K-W=SepalLengthCm SepalWidthCm PetalLengthCm PetalWidthCm BY SpeciesNumber(0 2)
  /MISSING ANALYSIS
  /METHOD=EXACT TIMER(5).


*Nonparametric Tests: Independent Samples. 
NPTESTS 
  /INDEPENDENT TEST (SepalLengthCm SepalWidthCm PetalLengthCm PetalWidthCm) GROUP (SpeciesNumber) 
  /MISSING SCOPE=ANALYSIS USERMISSING=EXCLUDE
  /CRITERIA ALPHA=0.05  CILEVEL=95.