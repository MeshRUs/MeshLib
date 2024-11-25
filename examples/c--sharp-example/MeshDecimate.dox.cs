using MR.DotNet;
using System;
using System.Reflection;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = Mesh.FromAnySupportedFormat( "mesh.stl" );

            // Setup decimate parameters
            DecimateParameters dp = new DecimateParameters();
            dp.strategy = DecimateStrategy.MinimizeError;
            dp.maxError = 1e-5f * mesh.BoundingBox.Diagonal();
            dp.tinyEdgeLength = 1e-3f;
            dp.packMesh = true;

            // Decimate mesh
            var result = MeshDecimate.Decimate(mesh, dp);

            // Save result
            Mesh.ToAnySupportedFormat(mesh, "decimated_mesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
