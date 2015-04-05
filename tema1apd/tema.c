/* Mangea Liviu Darius 334CA */

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(int args, char** argv) {
	
	char map_type;
	int w_harta, h_harta, w, h;
	int i, j, m;
	int maxw = 0, maxh = 0;

	omp_set_num_threads(atoi(argv[1]));
	
	FILE *fr, *fw;
	
	fr = fopen(argv[3], "r");	
	fw = fopen(argv[4], "w");
	
	fscanf(fr, "%c %d %d %d %d", &map_type, &w_harta, &h_harta, &w, &h);
	
	int harta[w_harta][h_harta], simulare[w][h], temp[w][h];
	
	/* citire harta din fisier */
	for(i = 0; i < w_harta; i++) {
		for(j = 0; j < h_harta; j++) {
			fscanf(fr, "%d", &harta[i][j]);
		}
	}
	fclose(fr);

	/* copiez harta in matricea simulare */
	#pragma omp parallel for schedule(static) private(i, j)
	for(i = 0; i < ((w < w_harta) ? w : w_harta); i++) {
		for(j = 0 ; j < ((h < h_harta) ? h : h_harta); j++) {
			simulare[i][j] = harta[i][j];
		}
	}
	
	
	/* for cu numarul de pasi din joc */
	for(m = 0; m < atoi(argv[2]); m++) {
	
		/* copiez simularea intr-o matrice temporara */
		#pragma omp parallel for schedule(static) private(i, j) 
		for(i = 0; i < w; i++) {
			for(j = 0; j < h; j++) {
				temp[i][j] = simulare[i][j];	
			}
		}
		
		/* calculez numarul de vecini pentru celulele ce nu sunt pe margine */
		#pragma omp parallel for schedule(static) private(i, j)
		for(i = 1; i < w - 1; i++) {
			for(j = 1; j < h - 1; j++) {
				simulare[i][j] = temp[i-1][j-1] + temp[i-1][j] + temp[i-1][j+1] + 
					temp[i][j-1] + temp[i][j+1] + temp[i+1][j-1] + temp[i+1][j] + 
					temp[i+1][j+1];
			}
		}
		
		/* calculez numarul de vecini pentru margini in functie de tipul 
		matricei de simulat, plan sau toroid */
		if(map_type == 'P') {
			#pragma omp parallel for private(j) collapse(1)
			for(j = 1; j < h - 1; j++) {
				simulare[0][j] = temp[0][j-1] + temp[0][j+1] + temp[1][j-1] + 
					temp[1][j] + temp[1][j+1];
				simulare[w-1][j] = temp[w-2][j-1] + temp[w-2][j+1] + temp[w-2][j] + 
					temp[w-1][j-1] + temp[w-1][j+1];
			}
			
			#pragma omp parallel for private(i) collapse(1)
			for(i = 1; i < w - 1; i++) {
				simulare[i][0] = temp[i][1] + temp[i-1][1] + temp[i+1][1] + 
					temp[i-1][0] + temp[i+1][0];
				simulare[i][h-1] = temp[i-1][h-2] + temp[i][h-2] + temp[i+1][h-2]
					+ temp[i-1][h-1] + temp[i+1][h-1];
			}
			
			simulare[0][0] = temp[0][1] + temp[1][0] + temp[1][1];
			simulare[w-1][0] = temp[w-2][1] + temp[w-2][0] + temp[w-1][1];
			simulare[0][h-1] = temp[0][h-2] + temp[1][h-2] + temp[1][h-1];
			simulare[w-1][h-1] = temp[w-1][h-2] + temp[w-2][h-1] + temp[w-2][h-2];	
		
		} else {
			
			#pragma omp parallel for private(j) collapse(1)
			for(j = 1; j < h - 1; j++) {
				simulare[0][j] = temp[w-1][j-1] + temp[w-1][j] + temp[w-1][j+1] + 
					temp[0][j-1] + temp[0][j+1] + temp[1][j-1] + temp[1][j] + 
					temp[1][j+1];
				simulare[w-1][j] = temp[0][j-1] + temp[0][j] + temp[0][j+1] + 
					temp[w-1][j-1] + temp[w-1][j+1] + temp[w-2][j-1] + temp[w-2][j] + 
					temp[w-2][j+1];
			}
			
			#pragma omp parallel for private(i) collapse(1)
			for(i = 1; i < w - 1; i++) {
				simulare[i][0] = temp[i-1][h-1] + temp[i][h-1] + temp[i+1][h-1] + 
					temp[i][1] + temp[i-1][1] + temp[i+1][1] + temp[i-1][0] + 
					temp[i+1][0];
				simulare[i][h-1] = temp[i-1][0] + temp[i][0] + temp[i+1][0] + 
					temp[i-1][h-2] + temp[i][h-2] + temp[i+1][h-2] + temp[i-1][h-1] + 
					temp[i+1][h-1];
			}
			
			simulare[0][0] = temp[0][1] + temp[1][0] + temp[1][1] + temp[w-1][h-1] +
				temp[0][h-1] + temp[w-1][0] + temp[1][h-1] + temp[w-1][1];
			simulare[w-1][0] = temp[w-2][1] + temp[w-2][0] + temp[w-1][1] + temp[0][0] +
				temp[w-1][h-1] + temp[0][h-1] + temp[0][1] + temp[w-2][h-1];
			simulare[0][h-1] = temp[0][h-2] + temp[1][h-2] + temp[1][h-1] + temp[w-1][0] +
				temp[0][0] + temp[w-1][h-1] + temp[w-1][h-2] + temp[1][0];
			simulare[w-1][h-1] = temp[w-1][h-2] + temp[w-2][h-1] + temp[w-2][h-2] + temp[0][0] +
				temp[w-1][0] + temp[0][h-1] + temp[0][h-2] + temp[w-2][0];
		}

		/* inlocuiesc in simulare cu 0 sau 1 in functie de numarul de vecini si 
		copiez valoarea si in matricea temp */
		#pragma omp parallel for schedule(static) private(i, j)
		for(i = 0; i < w; i++) {
			for(j = 0; j < h; j++) {
				if(temp[i][j] == 0) {
					if(simulare[i][j] == 3) {
						simulare[i][j] = 1;
					} else {
						simulare[i][j] = 0;
					}
				} else {
					if(simulare[i][j] == 2 || simulare[i][j] == 3) {
						simulare[i][j] = 1;
					} else {
						simulare[i][j] = 0;
					}
				}
			}
		}
	
 	}

	/* calculez ultima linie si ultima coloana unde am 1, pentru a afisa doar
	acea bucata */
	#pragma omp parallel for schedule(static) private(i, j) 
	for(i = 0; i <= w - 1; i++) {
		for(j = 0; j <= h - 1; j++) {
			if(simulare[i][j] == 1) {
				if(maxw < i) {
					maxw = i;
				} 
				if(maxh < j) {
					maxh = j;
				}
			}
		}
	}
	
	maxw++;
	maxh++;
	
	fw = fopen(argv[4], "w");

	fprintf(fw, "%c %d %d %d %d\n", map_type, maxh, maxw, w, h);
	
	/* scriu in fisier */
	for(i = 0; i < maxw; i++) {
		for(j = 0; j < maxh; j++) {
			fprintf(fw, "%d ", simulare[i][j]);
		}
		fprintf(fw, "\n");
	}
	
	fclose(fw);
	
	return 0;
}
