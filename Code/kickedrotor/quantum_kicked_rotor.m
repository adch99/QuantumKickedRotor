function main()
    F = floquet_generator(1000, 5, 1);
    %writematrix(F, "../quantum_kicked_rotor_floquet_matrix_matlab.txt", "Delimiter", " ");
    eigvals = eig(F);
    mods = abs(eigvals);
    phases = sort(angle(eigvals));
    diffs = phases(2:end) - phases(1:end-1);
    hist(diffs);
end



function F = floquet_generator(N, k, hbar)
    n = -N:N;
    [colgrid, rowgrid] = meshgrid(n, n);
    F = exp((-i * hbar / 2) * colgrid.^2 ) ...
        .* besselj(colgrid - rowgrid, -k/hbar) ...
        .* i.^(colgrid - rowgrid);
end
