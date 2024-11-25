using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static MR.DotNet.Vector3f;

namespace MR.DotNet
{   
    public enum MultipleEdgesResolveMode
    {
        None = 0, //does not avoid multiple edges
        Simple, //avoids creating edges that already exist in topology
        Strong //makes additional efforts to avoid creating multiple edges
    };

    //TODO: when Laplacian is implemented, move this enum there
    public enum EdgeWeights
    {
        /// all edges have same weight=1
        Unit,
        /// edge weight depends on local geometry and uses cotangent values
        Cotan,
        /// [deprecated] edge weight is equal to edge length times cotangent weight
        CotanTimesLength,
        /// cotangent edge weights and equation weights inversely proportional to square root of local area
        CotanWithAreaEqWeight
    }

    /** \struct MRFillHoleParams
     * \brief Parameters structure for FillHole\n
     * Structure has some options to control FillHole
     */
    public struct FillHoleParams
    {       
        /** Specifies triangulation metric\n
          * default for FillHole: GetCircumscribedFillMetric\n
          */
        public FillHoleMetric Metric = new FillHoleMetric();

        /// If not null accumulate new faces
        public BitSet? OutNewFaces = null;

        /** If Strong makes additional efforts to avoid creating multiple edges
          *
          * If Simple avoids creating edges that already exist in topology (default)
          *
          * If None does not avoid multiple edges
          */
        public MultipleEdgesResolveMode MultipleEdgesResolveMode = MultipleEdgesResolveMode.Simple;

        /** If true creates degenerate faces band around hole to have sharp angle visualization
          * \warning This flag bad for result topology, most likely you do not need it
          */
        public bool MakeDegenerateBand = false;

        /** The maximum number of polygon subdivisions on a triangle and two smaller polygons,
          * must be 2 or larger
          */
        public int MaxPolygonSubdivisions = 20;

        /** Input/output value, if it is present:
          * returns true if triangulation was bad and do not actually fill hole,
          * if triangulation is ok returns false;
          * if it is not present fill hole trivially in case of bad triangulation, (or leaves bad triangulation, depending on metric)
          */
        public bool? StopBeforeBadTriangulation = null;

        public FillHoleParams() {}
    };

    public struct FillHoleNicelyParams
    {
        public FillHoleParams triangulationParams = new FillHoleParams();
        /// If false then additional vertices are created inside the patch for best mesh quality
        public bool TriangulateOnly = false;
        ///Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
        public float MaxEdgeLen = 0;
        ///Maximum number of edge splits allowed during subdivision
        public int MaxEdgeSplits = 1000;
        ///Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
        public float MaxAngleChangeAfterFlip = 30.0f * (float)Math.PI / 180.0f;
        /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
        public bool SmoothCurvature = true;
        /// Additionally smooth 3 layers of vertices near hole boundary both inside and outside of the hole
        public bool NaturalSmooth = false;
        /// Edge weighting scheme for smoothCurvature mode
        public EdgeWeights EdgeWeights;
        public FillHoleNicelyParams() {}
    }

    public class MeshFillHole
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRFillHoleParams
        {
            public IntPtr metric = IntPtr.Zero;
            public IntPtr outNewFaces = IntPtr.Zero;
            public MultipleEdgesResolveMode multipleEdgesResolveMode = MultipleEdgesResolveMode.Simple;
            public byte makeDegenerateBand = 0;
            public int maxPolygonSubdivisions = 20;
            public IntPtr stopBeforeBadTriangulation = IntPtr.Zero;
            public MRFillHoleParams () {}
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRFillHoleNicelyParams
        {
            public MRFillHoleParams triangulationParams = new MRFillHoleParams();
            public byte triangulateOnly = 0;
            public float maxEdgeLen = 0;
            public int maxEdgeSplits = 1000;
            public float maxAngleChangeAfterFlip = 30.0f * (float)Math.PI / 180.0f;
            public byte smoothCurvature = 1;
            public byte naturalSmooth = 0;
            public EdgeWeights edgeWeights;
            public MRFillHoleNicelyParams() { }
        };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFillHole(IntPtr mesh, EdgeId a, ref MRFillHoleParams parameters );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFillHoles(IntPtr mesh, IntPtr pAs, ulong asNum, ref MRFillHoleParams parameters );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFillHoleNicely(IntPtr mesh, EdgeId holeEdge, ref MRFillHoleNicelyParams parameters );

        /** \brief Fills hole in mesh\n
          *
          * Fills given hole represented by one of its edges (having no valid left face),
          * default metric: CircumscribedFillMetric
          *
          * \param mesh mesh with hole
          * \param a EdgeId which represents hole (should not have valid left FaceId)
          * \param parameters parameters of hole filling
          *
          */
        unsafe public static void FillHole(ref Mesh mesh, EdgeId a, FillHoleParams parameters)
        {
            MRFillHoleParams mrParam;
            mrParam.metric = parameters.Metric.mrMetric_;
            mrParam.outNewFaces = parameters.OutNewFaces?.bs_ ?? IntPtr.Zero;
            mrParam.multipleEdgesResolveMode = parameters.MultipleEdgesResolveMode;
            mrParam.makeDegenerateBand = parameters.MakeDegenerateBand ? (byte)1 : (byte)0;
            mrParam.maxPolygonSubdivisions = parameters.MaxPolygonSubdivisions;
            
            byte stopBeforeBadTriangulation = 0;
            mrParam.stopBeforeBadTriangulation = parameters.StopBeforeBadTriangulation.HasValue ? new IntPtr( &stopBeforeBadTriangulation) : IntPtr.Zero;

            mrFillHole(mesh.varMesh(), a, ref mrParam);

            if (parameters.StopBeforeBadTriangulation.HasValue )
            {
                parameters.StopBeforeBadTriangulation = stopBeforeBadTriangulation > 0;
                if ( parameters.StopBeforeBadTriangulation.Value )
                    throw new Exception("Bad triangulation");
            }
        }

        /// fills a hole in mesh specified by one of its edge,
        /// optionally subdivides new patch on smaller triangles,
        /// optionally make smooth connection with existing triangles outside the hole
        /// \return triangles of the patch
        unsafe public static FaceBitSet FillHoleNicely(ref Mesh mesh, EdgeId holeEdge, FillHoleNicelyParams parameters)
        {
            MRFillHoleNicelyParams mrParam;
            mrParam.triangulationParams.metric = parameters.triangulationParams.Metric.mrMetric_;
            mrParam.triangulationParams.outNewFaces = parameters.triangulationParams.OutNewFaces?.bs_ ?? IntPtr.Zero;
            mrParam.triangulationParams.multipleEdgesResolveMode = parameters.triangulationParams.MultipleEdgesResolveMode;
            mrParam.triangulationParams.makeDegenerateBand = parameters.triangulationParams.MakeDegenerateBand ? (byte)1 : (byte)0;
            mrParam.triangulationParams.maxPolygonSubdivisions = parameters.triangulationParams.MaxPolygonSubdivisions;
            
            byte stopBeforeBadTriangulation = 0;
            mrParam.triangulationParams.stopBeforeBadTriangulation = parameters.triangulationParams.StopBeforeBadTriangulation.HasValue ? new IntPtr(&stopBeforeBadTriangulation) : IntPtr.Zero;

            mrParam.triangulateOnly = parameters.TriangulateOnly ? (byte)1 : (byte)0;
            mrParam.maxEdgeLen = parameters.MaxEdgeLen;
            mrParam.maxEdgeSplits = parameters.MaxEdgeSplits;
            mrParam.maxAngleChangeAfterFlip = parameters.MaxAngleChangeAfterFlip;
            mrParam.smoothCurvature = parameters.SmoothCurvature ? (byte)1 : (byte)0;
            mrParam.naturalSmooth = parameters.NaturalSmooth ? (byte)1 : (byte)0;
            mrParam.edgeWeights = parameters.EdgeWeights;

            var res = new FaceBitSet( mrFillHoleNicely(mesh.varMesh(), holeEdge, ref mrParam) );

            if (parameters.triangulationParams.StopBeforeBadTriangulation.HasValue)
            {
                parameters.triangulationParams.StopBeforeBadTriangulation = stopBeforeBadTriangulation > 0;
                if (parameters.triangulationParams.StopBeforeBadTriangulation.Value)
                    throw new Exception("Bad triangulation");
            }

            return res;
        }

        /// fill all holes given by their representative edges in \param edges
        unsafe public static void FillHoles( ref Mesh mesh, List<EdgeId> edges, FillHoleParams parameters )
        {
            MRFillHoleParams mrParam;
            mrParam.metric = parameters.Metric.mrMetric_;
            mrParam.outNewFaces = parameters.OutNewFaces?.bs_ ?? IntPtr.Zero;
            mrParam.multipleEdgesResolveMode = parameters.MultipleEdgesResolveMode;
            mrParam.makeDegenerateBand = parameters.MakeDegenerateBand ? (byte)1 : (byte)0;
            mrParam.maxPolygonSubdivisions = parameters.MaxPolygonSubdivisions;

            byte stopBeforeBadTriangulation = 0;
            mrParam.stopBeforeBadTriangulation = parameters.StopBeforeBadTriangulation.HasValue ? new IntPtr(&stopBeforeBadTriangulation) : IntPtr.Zero;

            int sizeOfEdgeId = Marshal.SizeOf(typeof(EdgeId));
            IntPtr nativeEdges = Marshal.AllocHGlobal(edges.Count * sizeOfEdgeId);

            try
            {                
                for ( int i = 0; i < edges.Count; ++i)
                {
                    Marshal.StructureToPtr(edges[i], IntPtr.Add(nativeEdges, i * sizeOfEdgeId), false);
                }

                mrFillHoles(mesh.varMesh(), nativeEdges, (ulong)edges.Count, ref mrParam);

                if (parameters.StopBeforeBadTriangulation.HasValue)
                {
                    parameters.StopBeforeBadTriangulation = stopBeforeBadTriangulation > 0;
                    if (parameters.StopBeforeBadTriangulation.Value)
                        throw new Exception("Bad triangulation");
                }
            }
            finally
            {
                Marshal.FreeHGlobal(nativeEdges);
            }
        }
    }
}
