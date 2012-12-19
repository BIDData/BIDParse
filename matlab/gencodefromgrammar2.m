nrules = size(udarr,1);

fh = fopen([codeprefix codetype 'callall.h'],'wt');
fprintf(fh,'#include <cuda_runtime.h>\n\n');
fprintf(fh,'const int %srules=%d;\n', codetype, nrules);
fprintf(fh,'const int %snsyms=%d;\n', codetype, nnsymbols);
fprintf(fh,'const int %stsyms=%d;\n', codetype, ntsymbols);
fprintf(fh,'const int %slens[]={%d,%d,%d};\n', codetype, lens);
fprintf(fh,'const int %snbs[]={%d,%d,%d};\n\n', codetype, nbs);
fprintf(fh,['extern int ' codetype 'callall(float *groot, float *left, float *right, float *scale, int ndo, int nthreads);\n\n']);
for ix = 1:lens(1)
  for iy = 1:lens(2)
    for iz = 1:lens(3)
      fprintf(fh,'__global__ extern void %srules_%d_%d_%d(float *groot, float *left, float *right, float *scale, int ndo);\n',codetype,ix,iy,iz);
    end
  end
end
fclose(fh);

fa = fopen([codeprefix codetype 'callall.cu'],'wt');
fprintf(fa,'#include <stdio.h>\n');
fprintf(fa,['#include "' codetype 'callall.h"\n\n']);
fprintf(fa,['int ' codetype 'callall(float *groot, float *left, float *right, float *scale, int ndo, int nthreads) {\n']);
fprintf(fa,'  cudaError_t err;\n');

regmat = zeros(prod(lens), prod(nbs));

fprintf('Generating code');
for ix = 1:lens(1)
  ixi = find(fassign(:,1) == ix);
  for iy = 1:lens(2)
    iyi = ixi(find(fassign(ixi,2) == iy));
    fprintf('.');
    for iz = 1:lens(3)
      ikernel = iz + lens(3)*(iy-1 + lens(2)*(ix-1));
      izi = iyi(find(fassign(iyi,3) == iz));
      fc = fopen([codeprefix codetype sprintf('rules_%d_%d_%d.cu',ix,iy,iz)],'wt');
      %      fc = fopen([codeprefix codetype sprintf('dmy.txt')],'wt');

      fprintf(fc,'#include <cuda_runtime.h>\n');
      fprintf(fc,['#include "testSparse.h"\n']);
      fprintf(fc,['#include "' codetype 'callall.h"\n\n']);
      fprintf(fc,'__global__ void %srules_%d_%d_%d(float *groot, float *left, float *right, float *scale, int ndo) {\n',codetype,ix,iy,iz);
      fprintf(fa,'  %srules_%d_%d_%d<<<%d,nthreads>>>(groot, left, right, scale, ndo);\n',codetype,ix,iy,iz,prod(nbs));
      fprintf(fa, '  cudaDeviceSynchronize();\n');
      fprintf(fa, '  err = cudaGetLastError();\n');
      fprintf(fa, '  if (err != cudaSuccess) {fprintf(stderr, "cuda error in %srules_%d_%d_%d"); return err;}\n',codetype,ix,iy,iz);

      fprintf(fc, '  switch (blockIdx.x) {\n');
      izimap = accumarray(izi, 1:length(izi), [size(fassign,1) 1]);
      for ibx = 1:nbs(1)
        ibxi = izi(find(fassign(izi,4) == ibx));
        for iby = 1:nbs(2)
          ibyi = ibxi(find(fassign(ibxi,5) == iby));
          for ibz = 1:nbs(3)
            icurr = ibyi(find(fassign(ibyi,6) == ibz));
            nbrules = length(icurr);
            ib = ibz-1 + nbs(3)*(iby-1 + nbs(2)*(ibx-1));
            fprintf(fc, '  case %d : {\n', ib);
            dcurr0 = udarr(icurr,:);          
            vcurr0 = udval(icurr,1);
            %[dmy ii] = sortrows([simorder(dcurr0(:,[2,3,1])) dcurr0(:,3) dcurr0(:,1)]);   % Group LR
            [dmy ii] = sortrows([dcurr0(:,2) dcurr0(:,3) dcurr0(:,1)]);   % Group LR pairs together
            dcurr = dcurr0(ii,:) - 1;  % Get zero-based indices for C
            vcurr = vcurr0(ii,1);
            [puniq dmy pmap] = unique(dcurr(:,1));
            [luniq dmy lmap] = unique(dcurr(:,2));
            [runiq dmy rmap] = unique(dcurr(:,3));
            
            fprintf(fc, ['    for (int tid = threadIdx.x; tid < ndo; tid += blockDim.x) {\n']);
            fprintf(fc,'      float ss = scale[tid];\n');
            % for j = 1:length(luniq);
            %   fprintf(fc, '      float L%03d = left[tid + %d * stride];\n', j, luniq(j));
            % end
            for j = 1:length(runiq);
              %              fprintf(fc, '      float R%03d = right[tid + %d * stride];\n', j, runiq(j));
            end
            pdone = zeros(length(puniq),1);
            lastl = -1;
            irev = (nbrules:-1:1)';
            ifwd = (1:nbrules)';
            pall = accumarray([ifwd pmap], 1, [nbrules, length(puniq)]);
            prev = cumsum(pall(irev,:));
            ptimes = (cumsum(pall)>0) & (prev(irev,:)>0);
            lall = accumarray([ifwd lmap], 1, [nbrules, length(luniq)]);
            lrev = cumsum(lall(irev,:));
            ltimes = (cumsum(lall)>0) & (lrev(irev,:)>0);
            rall = accumarray([ifwd rmap], 1, [nbrules, length(runiq)]);
            rrev = cumsum(rall(irev,:));
            rtimes = (cumsum(rall)>0) & (rrev(irev,:)>0);
            regmat(ikernel, ib+1) = max(0, max(sum(ptimes, 2) + sum(ltimes,2) + sum(rtimes,2)));
            [dmy prii] = sort(sum(pall)');
            reload = accumarray(prii(1:(end-max(0,(50-max(sum(rtimes,2))))),1), 1, [size(pall,2) 1]);
            
            for j = 1:nbrules
              if (ltimes(j, lmap(j)) == 1 && (j == 1 || ltimes(j-1, lmap(j)) == 0))
                fprintf(fc, '      float L%03d = left[tid + %d * stride];\n', lmap(j), luniq(lmap(j)));
              end
              if (rtimes(j, rmap(j)) == 1 && (j == 1 || rtimes(j-1, rmap(j)) == 0))
                fprintf(fc, '      float R%03d = right[tid + %d * stride];\n', rmap(j), runiq(rmap(j)));
              end
                if (pdone(pmap(j)))
                  fprintf(fc, '      G%03d = G%03d + L%03d * R%03d * %7.6ef;\n', pmap(j), pmap(j), lmap(j), rmap(j), vcurr(j));
                  % fprintf(fc, '      G%03d = max(G%03d, L%03d * R%03d * %7.6ef);\n', pmap(j), pmap(j), lmap(j), rmap(j), vcurr(j));
                else
                  fprintf(fc, '      float G%03d = L%03d * R%03d * %7.6ef;\n', pmap(j), lmap(j), rmap(j), vcurr(j));                      
                  pdone(pmap(j)) = 1;
                end
                if (ptimes(j, pmap(j)) == 1 && (j == nbrules || ptimes(j+1, pmap(j)) == 0))
                  fprintf(fc, '      atomicAdd(&groot[tid + %d * stride], ss * G%03d);\n', puniq(pmap(j)), pmap(j));
                  %                  fprintf(fc, '      G%03d *= ss;\n', pmap(j));                  
                  %                  fprintf(fc, '      atomicMax((int *)&groot[tid + %d * stride], *((int *)&G%03d));\n', puniq(pmap(j)), pmap(j));
                end
                %              end
            end
            for jp = 1:length(puniq);
              %              fprintf(fc, '      atomicAdd(&groot[tid + %d * stride], G%03d * ss);\n', puniq(jp), jp);
            end
            fprintf(fc, '    }\n  } break;\n');
          end
        end
      end
      fprintf(fc, '  }\n}\n');
      fclose(fc);
    end
  end
end
fprintf(fa,'  return 0;\n}\n');
fprintf('\n');
fclose(fa);

fd = fopen([dataprefix codetype 'binrulesx.dat'],'wb');
fwrite(fd,nrules,'int32');
fwrite(fd,udarr(:)-1,'int32');
fwrite(fd,udval(:),'float');
fclose(fd);
