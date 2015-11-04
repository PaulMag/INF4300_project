%%
close all

%% 4 at a time
subplot(2,2,1)
imshow(g1d01 / max(max(g1d01)))
subplot(2,2,2)
imshow(g2d01 / max(max(g2d01)))
subplot(2,2,3)
imshow(g3d01 / max(max(g3d01)))
subplot(2,2,4)
imshow(g4d01 / max(max(g4d01)))

%% All 16 at a time
% Each row represent one direction.
% Each column represent one texture.

% dx=+1, dy= 0
subplot(4,4,1)
imshow(g1d01 / max(max(g1d01)))
subplot(4,4,2)
imshow(g2d01 / max(max(g2d01)))
subplot(4,4,3)
imshow(g3d01 / max(max(g3d01)))
subplot(4,4,4)
imshow(g4d01 / max(max(g4d01)))

% dx= 0, dy=-1
subplot(4,4,5)
imshow(g1d10 / max(max(g1d10)))
subplot(4,4,6)
imshow(g2d10 / max(max(g2d10)))
subplot(4,4,7)
imshow(g3d10 / max(max(g3d10)))
subplot(4,4,8)
imshow(g4d10 / max(max(g4d10)))

% dx=+1, dy=-1
subplot(4,4, 9)
imshow(g1d1min1 / max(max(g1d1min1)))
subplot(4,4,10)
imshow(g2d1min1 / max(max(g2d1min1)))
subplot(4,4,11)
imshow(g3d1min1 / max(max(g3d1min1)))
subplot(4,4,12)
imshow(g4d1min1 / max(max(g4d1min1)))

% dx=-1, dy=-1
subplot(4,4,13)
imshow(g1dmin11 / max(max(g1dmin11)))
subplot(4,4,14)
imshow(g2dmin11 / max(max(g2dmin11)))
subplot(4,4,15)
imshow(g3dmin11 / max(max(g3dmin11)))
subplot(4,4,16)
imshow(g4dmin11 / max(max(g4dmin11)))


%%
subplot(4,4,15)
ylabel('')


%%


