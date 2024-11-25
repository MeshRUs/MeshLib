using MR.DotNet;
using System;
using System.Globalization;
using System.Reflection;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length != 2 && args.Length != 3)
        {           
            Console.WriteLine("Usage: {0} OFFSET_VALUE", Assembly.GetExecutingAssembly().GetName().Name);
            return;
        }

        try
        {
            float offsetValue = float.Parse(args[0],
                      System.Globalization.NumberStyles.AllowThousands,
                      CultureInfo.InvariantCulture);

            // Load mesh
            MeshPart mp = new MeshPart();
            mp.mesh = Mesh.FromAnySupportedFormat( "mesh.stl" );

            // Setup parameters
            OffsetParameters op = new OffsetParameters();
            op.voxelSize = Offset.SuggestVoxelSize(mp, 1e6f);

            // Make offset mesh
            var result = Offset.OffsetMesh(mp, offsetValue, op);

            // Save result
            Mesh.ToAnySupportedFormat(result, "mesh_offset.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
